import cloudpickle
import multiprocessing as mp
import torch
from utils.shape import build_voxel_shapeAE_model, build_point_shapeAE_model
from functools import partial

class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var):
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var):
        self.var = cloudpickle.loads(var)
        
def _worker(
    remote: mp.connection.Connection, parent_remote: mp.connection.Connection, init_fn_wrapper: CloudpickleWrapper
) -> None:
    parent_remote.close()
    shape_AE = init_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "encode":
                batch_zs = shape_AE.encoder(data).detach()
                remote.send(batch_zs)
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except Exception as error:
            print(error)
            break

class SubprocShapeAE:
    def __init__(self, config):
        self.waiting = False
        self.closed = False
        n_AEs = config.num_workers
        if config.shape_type == 'pointAE_shape':
            init_fn = partial(build_point_shapeAE_model, config)
        elif config.shape_type == 'IMAE_shape':
            init_fn = partial(build_voxel_shapeAE_model, config)
        else:
            raise ValueError("unidentified shape type: %s" % (config.shape_type))
        init_fns = [init_fn for _ in range(n_AEs)]

        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)
        
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_AEs)])
        self.processes = []
        for work_remote, remote, init_fn in zip(self.work_remotes, self.remotes, init_fns):
            args = (work_remote, remote, CloudpickleWrapper(init_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()
    
    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def encode_async(self, batch_voxels, num_batch):
        for remote, batch_voxel in zip(self.remotes[:num_batch], batch_voxels):
            remote.send(("encode", batch_voxel))
        self.waiting = True

    def encode_wait(self, num_batch):
        results = [remote.recv() for remote in self.remotes[:num_batch]]
        self.waiting = False
        if len(results) == 0: return None
        return torch.concat(results, 0)
    
    def encode(self, batch_voxels):
        num_batch = len(batch_voxels)
        self.encode_async(batch_voxels, num_batch)
        return self.encode_wait(num_batch)

