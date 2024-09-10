from coperception.models.det.base.IntermediateModelBase import IntermediateModelBase
import torch
import pickle
import json
import matplotlib.pyplot as plt
import os

class FusionBase(IntermediateModelBase):
    def __init__(
        self,
        config,
        layer=3,
        in_channels=13,
        kd_flag=True,
        num_agent=5,
        compress_level=0,
        only_v2i=False,
    ):
        super().__init__(config, layer, in_channels, kd_flag, num_agent, compress_level, only_v2i)
        self.num_agent = 0

    def fusion(self):
        raise NotImplementedError(
            "Please implement this method for specific fusion strategies"
        )

    def forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1, pert=None, eps=None, attacker_list=None, ego_agent=None, unadv_pert=None, kick=False, no_fuse=False, collab_agent_list=None, trial_agent_id=None, current_file_name=None, adv_method=None):

        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        encoded_layers = self.u_encoder(bevs)
        device = bevs.device


        if not no_fuse:
            # print("Fusion Activated")

            feat_maps, size = super().get_feature_maps_and_size(encoded_layers)

            # print("Feature Maps Shape: ", feat_maps.shape)

            # if pert is not None:
                # print("Perturbation is applied on agent {}".format(attacker_list))
                # clip
                # eta = torch.clamp(pert, min=-eps, max=eps)
                # Apply perturbation
                # feat_maps[attacker_list] = feat_maps[attacker_list] + eta
            # else:
            #     print("Perturbation is not applied")
                                    
            # if unadv_pert is not None:
            #     # print("Unadversarial perturbation is applied on agent 2")
            #     feat_maps[2] = feat_maps[2] + unadv_pert
            # else:
                # print("Unadversarial perturbation is not applied")

            feat_list = super().build_feature_list(batch_size, feat_maps)
            # print("feat_list length: ", len(feat_list))

            # for i, feat in enumerate(feat_list):
            #     print(f"Tensor {i} shape: {feat.shape}\n")  

            local_com_mat = super().build_local_communication_matrix(
                feat_list
            )  # [1 6 256 32 32] [batch, agent, channel, height, width]

            
            local_com_mat_update = super().build_local_communication_matrix(
                feat_list
            )  # to avoid the inplace operation

            

            for b in range(batch_size): 

                self.num_agent = num_agent_tensor[b, 0] # num_agent: the number of agents
                for i in range(self.num_agent): # 在一个batch里循环每一个agent
                    self.tg_agent = local_com_mat[b, i] # tg_agent is the current agent's feature map
                    self.neighbor_feat_list = []
                    self.neighbor_feat_list.append(self.tg_agent)
                    all_warp = trans_matrices[b, i]  # transformation [2 5 5 4 4]
                    # i == 1: ego agent
                    
                    # build neighbors feature list for each agent 
                    if ego_agent is not None and i == ego_agent: # 第i个agent是ego agent 

                        # 在这个条件语句下可以得到 neighbors_feature_list
                        super().build_neighbors_feature_list(
                            b,
                            i,
                            all_warp,
                            self.num_agent,
                            local_com_mat,
                            device,
                            size,
                            trans_matrices,
                            collab_agent_list, 
                            trial_agent_id,
                            pert,
                            attacker_list,
                            eps
                        )
                        self.attacked_feature_dict[i] = [self.tg_agent, 'ego']
                        # TODO:save attacked_feature_dict
                        # scene + frame + attack_name 
                        
                        if current_file_name:
                            save_dir = '/data2/user2/senkang/CP-GuardBench/CP-GuardBench_RawData/generated/'
                            os.makedirs(save_dir, exist_ok=True)
                            
                            file_name = current_file_name[0][0].split('/')[-2]
                            with open(os.path.join(save_dir, f'{file_name}.pkl'), 'wb') as f:
                                pickle.dump(self.attacked_feature_dict, f)
                            print(f"Attacked feature dict saved to {os.path.join(save_dir, file_name)}.pkl")


                            for agent_id, (feature_map, agent_type) in self.attacked_feature_dict.items():
                                # Average across channels
                                avg_feature = feature_map.mean(dim=0).cpu().numpy()
                                
                                # Create a new figure
                                fig = plt.figure(figsize=(10, 8))
                                plt.imshow(avg_feature, cmap='plasma')
                                plt.colorbar()
                                plt.title(f'Agent {agent_id} ({agent_type}) Feature Map')
                                
                                if agent_type == 'ego':
                                    agent_postfix = 'ego'
                                elif agent_type == 1:
                                    agent_postfix = adv_method
                                else:
                                    agent_postfix = 'normal'
                                # Save the figure without displaying
                                save_path = os.path.join(save_dir, f'{file_name}_agent{agent_id}_{agent_postfix}.png')
                                plt.savefig(save_path)
                                plt.close(fig)

                                print(f"Visualization saved to {save_path}")


                    else:
                        super().build_neighbors_feature_list(     
                            b,
                            i,
                            all_warp,
                            self.num_agent,
                            local_com_mat,
                            device,
                            size,
                            trans_matrices,
                        )

                    
                    


                    # feature update
                    local_com_mat_update[b, i] = self.fusion()   # 最终每个 agent 以其自身为 ego 的特征融合结果
            
            # weighted feature maps is passed to decoder
            feat_fuse_mat = super().agents_to_batch(local_com_mat_update)

        else:
            # print("Fusion disabled")
            feat_maps, size = super().get_feature_maps_and_size(encoded_layers)

            feat_list = super().build_feature_list(batch_size, feat_maps)
            
            local_com_mat = super().build_local_communication_matrix(
                feat_list
            )  # [2 5 512 16 16] [batch, agent, channel, height, width]
            feat_fuse_mat = super().agents_to_batch(local_com_mat)

        decoded_layers = super().get_decoded_layers(
            encoded_layers, feat_fuse_mat, batch_size
        )
        x = decoded_layers[0]

        cls_preds, loc_preds, result = super().get_cls_loc_result(x)

        if self.kd_flag == 1:
            return (result, *decoded_layers, feat_fuse_mat)
        else:
            return result
