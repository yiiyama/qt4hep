import numpy as np
from scipy.sparse import coo_array
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from .staggered_fermion import get_rapidity
from .sparse import dagger, jw_annihilator_spo, ab_to_phi_sparse
jax.config.update('jax_enable_x64', True)


def setup(num_sites, lsp, mass, l0):
    half_lat = num_sites // 2
    hdim = 2 ** num_sites
    rapidity, wavenumber = get_rapidity(num_sites, lsp, mass, with_wn=True)

    print('Identifying Fock-space physical state indices')
    fock_config = (jnp.arange(hdim)[:, None] >> jnp.arange(num_sites)[None, :]) % 2
    sign = jnp.repeat(np.array([1, -1]), half_lat)
    charge = jnp.sum(fock_config * sign[None, :], axis=1)
    fock_indices = np.nonzero(charge == 0)[0]
    subdim = fock_indices.shape[0]

    print('Free Hamiltonian')
    occupancy = fock_config[fock_indices]
    energy = mass * np.cosh(rapidity)
    energy = np.tile(energy, 2)
    h_free = jnp.sum(energy[None, :] * occupancy, axis=1)

    print('Identifying position-space physical state indices')
    fermion_config = fock_config.copy()
    total_excitations = jnp.sum(fermion_config, axis=1)
    pos_indices = np.nonzero(total_excitations == half_lat)[0]

    print('Constructing position-space number operators')
    fock_ab = [op.to_matrix(sparse=True) for op in jw_annihilator_spo(num_sites)]
    phi = ab_to_phi_sparse(fock_ab, rapidity, wavenumber)
    fermi_num_coords = []
    fermi_num_data = []
    for op in phi:
        op = op[:, fock_indices]
        arr = coo_array(dagger(op) @ op)
        fermi_num_coords.append(np.array(arr.coords).T)
        fermi_num_data.append(arr.data)
    fermi_num = BCOO((jnp.array(fermi_num_data), jnp.array(fermi_num_coords)),
                     shape=(num_sites, subdim, subdim))
    print('Computing basis change matrix')
    basis_change_mat = position_states_as_fock_state_sums(pos_indices, fermi_num)

    print('Computing the electric Hamiltonian')
    charge = jnp.tile(np.array([0, 1]), half_lat)[None, :] - fermion_config[pos_indices]
    electric_config = jnp.cumsum(charge, axis=1) + l0
    l2 = jnp.sum(jnp.square(electric_config), axis=1)

    h_elec = jnp.einsum('h,ih,jh->ij', l2, basis_change_mat, basis_change_mat.conjugate())
    h_elec = jnp.where(jnp.isclose(h_elec, 0.), 0., h_elec)

    return fock_indices, pos_indices, fermi_num, basis_change_mat, h_free, h_elec


@jax.jit
def _update_proj(isite, data):
    proj, bidx, fermi_num = data
    proj = jax.lax.cond(
        jnp.equal(bidx[isite], 1),
        lambda: fermi_num[isite] @ proj,
        lambda: proj - fermi_num[isite] @ proj
    )
    return (proj, bidx, fermi_num)


@jax.jit
def _position_as_fock(idx, fermi_num, proj_init):
    bidx = (idx >> jnp.arange(fermi_num.shape[0])) % 2
    proj = jax.lax.fori_loop(
        1, fermi_num.shape[0], _update_proj, (proj_init[bidx[0]], bidx, fermi_num)
    )[0]
    return jnp.linalg.eigh(proj)[1][:, -1]

    # absvals = jnp.sqrt(jnp.diagonal(proj).real)
    # ikey = jax.lax.while_loop(
    #     lambda i: jnp.isclose(absvals[i], 0.),
    #     lambda i: i + 1,
    #     0
    # )
    # return absvals * jnp.exp(1.j * jnp.angle(proj[:, ikey]))


@jax.jit
def _position_as_fock_i(icol, data):
    mat, pos_indices, fermi_num, proj_init = data
    mat = mat.at[:, icol].set(_position_as_fock(pos_indices[icol], fermi_num, proj_init))
    return (mat, pos_indices, fermi_num, proj_init)


# _position_as_fock_v = jax.jit(jax.vmap(_position_as_fock, in_axes=(0, None, None), out_axes=1))


@jax.jit
def position_states_as_fock_state_sums(pos_indices, fermi_num):
    pos_indices = jnp.asarray(pos_indices)
    subdim = pos_indices.shape[0]

    proj_init = jnp.array([
        jnp.eye(subdim, dtype=fermi_num.dtype) - fermi_num[0].todense(),
        fermi_num[0].todense()
    ])

    mat_init = jnp.empty((subdim, subdim), dtype=fermi_num.dtype)
    return jax.lax.fori_loop(0, subdim, _position_as_fock_i,
                             (mat_init, pos_indices, fermi_num, proj_init))[0]

    # batch_size = 32
    # num_batches = subdim // batch_size + int(subdim % batch_size != 0)
    # batched_subdim = batch_size * num_batches
    # pos_indices_batched = jnp.reshape(
    #     jnp.concatenate([
    #         pos_indices,
    #         jnp.zeros(batched_subdim - subdim, dtype=pos_indices.dtype)
    #     ]),
    #     (num_batches, batch_size)
    # )

    # @jax.jit
    # def _position_as_fock_vi(ibatch, mat):
    #     return mat.at[:, ibatch].set(
    #         _position_as_fock_v(pos_indices_batched[ibatch], fermi_num, proj_init)
    #     )

    # mat_init = jnp.empty((subdim, num_batches, batch_size), dtype=fermi_num.dtype)
    # mat = jax.lax.fori_loop(0, num_batches, _position_as_fock_vi, mat_init)
    # return mat.reshape((subdim, batched_subdim))[:, :subdim]
