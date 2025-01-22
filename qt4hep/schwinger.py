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
    h_free = np.sum(energy[None, :] * occupancy, axis=1)

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
    h_elec = np.array(h_elec)

    return fock_indices, pos_indices, fermi_num, basis_change_mat, h_free, h_elec


def position_states_as_fock_state_sums(pos_indices, fermi_num):
    pos_indices = jnp.asarray(pos_indices)
    num_sites = fermi_num.shape[0]
    subdim = pos_indices.shape[0]

    fermi_num_0 = fermi_num[0].todense()
    fermi_num_0_compl = (-fermi_num_0).at[jnp.arange(subdim), jnp.arange(subdim)].add(1.)

    @jax.jit
    def occupied_0():
        return fermi_num_0

    @jax.jit
    def unoccupied_0():
        return fermi_num_0_compl

    @jax.jit
    def occupied(isite, proj):
        return fermi_num[isite] @ proj

    @jax.jit
    def unoccupied(isite, proj):
        compl = fermi_num[isite] @ proj
        return proj - compl

    @jax.jit
    def _position_as_fock(idx):
        bidx = (idx >> jnp.arange(num_sites)) % 2
        proj = jax.lax.cond(
            jnp.equal(bidx[0], 1),
            occupied_0,
            unoccupied_0
        )
        for isite in range(1, num_sites):
            proj = jax.lax.cond(
                jnp.equal(bidx[isite], 1),
                occupied,
                unoccupied,
                isite,
                proj
            )

        # return jnp.linalg.eigh(proj)[1][:, -1]

        absvals = jnp.sqrt(jnp.diagonal(proj).real)
        ikey = jax.lax.while_loop(
            lambda i: jnp.isclose(absvals[i], 0.),
            lambda i: i + 1,
            0
        )
        return absvals * jnp.exp(1.j * jnp.angle(proj[:, ikey]))

    @jax.jit
    def _position_as_fock_i(icol, mat):
        return mat.at[:, icol].set(_position_as_fock(pos_indices[icol]))

    return jax.lax.fori_loop(0, subdim, _position_as_fock_i,
                             jnp.empty((subdim, subdim), dtype=np.complex128))

    # _position_as_fock_v = jax.vmap(_position_as_fock, out_axes=1)

    # return _position_as_fock_v(pos_indices)

    # batch_size = 128
    # num_batches = subdim // batch_size - 1
    # mat = jnp.empty((subdim, subdim), dtype=np.complex128)
    # for ibatch in range(num_batches):
    #     start = batch_size * ibatch
    #     end = batch_size * (ibatch + 1)
    #     mat = mat.at[:, start:end].set(
    #         _position_as_fock_v(pos_indices[start:end])
    #     )
    # start = batch_size * num_batches
    # return mat.at[:, start:].set(_position_as_fock_v(pos_indices[start:]))
