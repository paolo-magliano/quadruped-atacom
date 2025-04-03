
import torch
import pytorch_kinematics as pk

class LinkPos(object):
    def __init__(self, urdf_path, end_effector, base, default_joint_angles, q_idx=None):
        self.dtype = default_joint_angles.dtype
        self.device = default_joint_angles.device

        self.q_idx = q_idx if q_idx is not None else list(range(default_joint_angles.shape[-1]))

        self.chain = pk.build_chain_from_urdf(open(urdf_path, mode="rb").read())

        base_chain = pk.SerialChain(self.chain, base).to(dtype=self.dtype, device=self.device)
        self.base_matrix = base_chain.forward_kinematics(default_joint_angles[self.q_idx][:len(base_chain.get_joint_parameter_names())]).get_matrix()

        self.end_effector_chain = pk.SerialChain(self.chain, end_effector).to(dtype=self.dtype, device=self.device)

    def get_matrix(self, q):
        self._check_dtype_device(q)
        return self.end_effector_chain.forward_kinematics(q[..., self.q_idx]).get_matrix() 

    def get_pos(self, q):
        return self.get_matrix(q)[:, :3, 3] - self.base_matrix[:, :3, 3]
    
    def get_rot(self, q):
        return self.get_matrix(q)[:, :3, :3] @ torch.inverse(self.base_matrix[:, :3, :3])
    
    def get_J(self, q):
        self._check_dtype_device(q)
        batch = q.shape[0] if len(q.shape) > 1 else 1
        J = torch.zeros((batch, 6, q.shape[-1])).to(q.device)
        J[:, :, self.q_idx] = self.end_effector_chain.jacobian(q[..., self.q_idx])
        return J
    
    def _check_dtype_device(self, q):
        if q.dtype != self.end_effector_chain.dtype or q.dtype != self.base_matrix.dtype or q.device != self.end_effector_chain.device or q.device != self.base_matrix.device:
            self.end_effector_chain = self.end_effector_chain.to(dtype=q.dtype, device=q.device)
            self.base_matrix = self.base_matrix.to(dtype=q.dtype, device=q.device)
    
        