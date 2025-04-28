import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import time
import tqdm
import os

def svd_lstsq(AA, BB):
    if BB.ndim==2:
        BB = BB.unsqueeze(-1)
    tol=1e-5
    U, S, Vh = torch.linalg.svd(AA, full_matrices=False)
    Spinv = torch.zeros_like(S)
    Spinv[S>tol] = 1/S[S>tol]
    UhBB = U.adjoint() @ BB
    if Spinv.ndim!=UhBB.ndim:
      Spinv = Spinv.unsqueeze(-1)
    SpinvUhBB = Spinv * UhBB
    return (Vh.adjoint() @ SpinvUhBB).squeeze(-1)

def worker(A, B, queue, rank):
    sol = torch.linalg.lstsq(A, B).solution
    queue.put((rank, sol))

def multiprocess_lstsq(A, B, N=2):
    N = min(N, A.shape[0])
    A_split = torch.chunk(A, N, dim=0)
    B_split = torch.chunk(B, N, dim=0)

    queue = mp.Queue()
    processes = []
    for i in range(N):
        p = mp.Process(target=worker, args=(A_split[i], B_split[i], queue, i))
        processes.append(p)
        p.start()

    results = [None] * N
    for i in range(N):
        rank, result = queue.get()
        results[rank] = result

    for p in processes:
        p.join()

    return torch.cat(results, dim=0)

def big_matrix_lstsq(A, B):
    A_big = torch.block_diag(*[A[i] for i in range(A.shape[0])])
    B_big = torch.cat([B[i] for i in range(B.shape[0])], dim=0)
    big_sol = torch.linalg.lstsq(A_big, B_big).solution
    sol = torch.stack([big_sol[i*A.shape[2]:(i+1)*A.shape[2]] for i in range(A.shape[0])])
    return sol
    
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(0)
    mp.set_start_method('spawn', force=True)

    path = os.path.dirname(os.path.realpath(__file__))

    vmap_lstsq = torch.vmap(lambda A, B: torch.linalg.lstsq(A, B.unsqueeze(-1)).solution.squeeze(-1))
    vmap_pseudo = torch.vmap(lambda A, B: torch.linalg.pinv(A) @ B)

    script_svd = torch.jit.script(svd_lstsq)

    compiled_vmap_lstsq = torch.compile(vmap_lstsq)
    compiled_lstsq = torch.compile(torch.linalg.lstsq)

    # Load matrixes
    # A shape (batch, n, m) B shape (batch, n) -> x shape (batch, m)
    # Where batch is 4096, n is 24, m is 36
    A = torch.load(os.path.join(path, "A.pt"))[:, 5:, 5:]
    B = torch.load(os.path.join(path, "B.pt"))[:, 5:]

    # Run the compiled functions once to compile them
    compiled_lstsq(A, B)
    compiled_vmap_lstsq(A, B) 

    # Make a list with the functions to test, every function should have the same signature (A, B)
    functions_to_test = {
        'lstsq': lambda A, B: torch.linalg.lstsq(A, B).solution,
        #'svd_lstsq': svd_lstsq,
        'vmap_lstsq': vmap_lstsq,
        # 'vmap_pseudo': vmap_pseudo,
        # 'script_svd': script_svd,
        # 'compiled_lstsq': lambda A, B: compiled_lstsq(A, B).solution,
        # 'compiled_vmap_lstsq': compiled_vmap_lstsq,
        'big_lstsq': big_matrix_lstsq,
    }
    times = {f: [] for f in functions_to_test}

    # Solution of the problem:
    solution = torch.linalg.lstsq(A, B).solution

    # Test all the batch sizes until max_batch_size (do not set over 4096)
    max_batch_size = 256

    # Measure the time for every batch size, from 1 to max_batch_size
    for i in tqdm.tqdm(range(1, max_batch_size)):
        for name, func in functions_to_test.items():
            start = time.time()
            x = func(A[:i], B[:i])
            # assert torch.allclose(x, solution[:i], atol=1e-7)
            times[name].append(time.time() - start)

    # Plot the results
    plt.figure()
    for name, time_list in times.items():
        plt.plot(time_list[5:], label=name, alpha=0.7)
    plt.legend()
    plt.xlabel('Batch size')
    plt.ylabel('Time (s)')

    plt.savefig('plot/lstsq/lstsq_comparison.png')
