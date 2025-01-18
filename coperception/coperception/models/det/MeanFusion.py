import torch
from coperception.models.det.base import FusionBase

from coperception.models.det.myModel import ResNetBinaryClassifier 


class MeanFusion(FusionBase):
    "Mean fusion. Used as a lower-bound in the DiscoNet fusion."

    def __init__(self, config, layer=3, in_channels=13, kd_flag=True, num_agent=5, compress_level=0, only_v2i=False):
        super().__init__(config, layer, in_channels, kd_flag, num_agent, compress_level, only_v2i)
        
    def fusion(self, cp_guard_defense=False):

        cp_guard = ResNetBinaryClassifier()
        pretrained_weights_path = '/data2/user2/senkang/CP-GuardBench/cpguard/logs/2024-09-09-17-06-32/49.pth'
        cp_guard.load_state_dict(torch.load(pretrained_weights_path))
        cp_guard.to('cuda')
        cp_guard.eval()

        ego_id = 3

        input = torch.stack(self.neighbor_feat_list).squeeze().to('cuda')
        outputs, _ = cp_guard(input)
        outputs = outputs.squeeze()

        # print(cp_guard_defense)
        if cp_guard_defense:
            for i in range(len(outputs)):
                if i == ego_id:
                    continue
                
                if outputs[i].item() > 0.5:
                    self.neighbor_feat_list[i] = torch.zeros_like(self.neighbor_feat_list[i])
                    print(f"Agent {i+1} is adversarial", outputs[i].item())
        
        return torch.mean(torch.stack(self.neighbor_feat_list), dim=0)  # 最终的特征融合
