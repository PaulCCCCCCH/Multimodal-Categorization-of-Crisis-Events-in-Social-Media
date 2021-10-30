# CVPR 2020: Multimodal Categorization of Crisis Events in Social Media

This is an unofficial implementation for the CVPR 2020 paper [*Multimodal Categorization of Crisis Events in Social Media*](https://openaccess.thecvf.com/content_CVPR_2020/papers/Abavisani_Multimodal_Categorization_of_Crisis_Events_in_Social_Media_CVPR_2020_paper.pdf).

> Abavisani, Mahdi, et al. "Multimodal categorization of crisis events in social media." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

To cite the paper:
```
@inproceedings{abavisani2020multimodal,
  title={Multimodal categorization of crisis events in social media},
  author={Abavisani, Mahdi and Wu, Liwei and Hu, Shengli and Tetreault, Joel and Jaimes, Alejandro},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14679--14689},
  year={2020}
}
```

## Note
This implementation follows the original paper whenever possible. Due to our urgent need for experiment results, we haven't had time to make it super configurable with clean handlers.


## To Run
- Initialize by running `bash setup.sh`
- Run the pipeline with `python main.py`

## Stats
We applied mixed-precision training, so it runs fast on GPUs with tensorcores (e.g. V100). The default configuration consumes about 13GB of GPU memory, and each epoch takes 3 minites on an Amazon `g4dn-xlarge` instance (with V100 GPU).

**Warning: Model is saved for each epoch, which means it consumes 400MB of disk every 3 minutes. Take this into consideration.**


## Confusions
### Equation 4
The authors stated that $$\alpha_{v_i}$$ was completely dependent on $$e_i$$, and $$\alpha_{e_i}$$ was completely dependent on $$\alpha_{v_i}$$, while the equations meant the opposite. The implementation will stick to the text instead of the equations.

### Self-Attention in Fully Connected Layers
After obtaining a multimodal representation that incorporates both visual and textual information, the authors used fully-connected layers to perform classification. Here the authors wrote 

> We add self-attention in the fully-connected networks. 

 We assumed that they meant 'we added a fully-connected layer as self-attention'.

### DenseNet
The authors did not give the size of the DenseNet they used.


## Todos
- T1: Check SSE implementation
- T1: Put configurations 
- T2: More reasonable model saving (priority queue, save n best)
- T2: Setting `num_workers > 1` deadlocks the dataloader.
- T3: Better logging





## Experiments 

### Task 1

|      | Arch V / L          | pv, pv0   | pt, pt0   | Dev acc | Train acc |      |      |
| ---- | ------------------- | --------- | --------- | ------- | --------- | ---- | ---- |
|      | DenseNet 201 / Bert | No SSE    | No SSE    | ~83     | 90+       |      |      |
|      | DenseNet 201 / Bert | 1000, 0.5 | 1000, 0.5 | ~68     | 90 +      |      |      |
|      | DenseNet 201 / Bert | 1000, 0.3 | 0.7       | ~83     | 90+       |      |      |
|      | DenseNet 201 / Bert | 1000, 0.5 | 0.98      | ~83     | 90+       |      |      |
|      |                     |           |           |         |           |      |      |
|      |                     |           |           |         |           |      |      |
|      |                     |           |           |         |           |      |      |
|      |                     |           |           |         |           |      |      |
|      |                     |           |           |         |           |      |      |

### Task 2

|      | Arch V / L          | pv, pv0    | pt, pt0    | Dev acc | Train acc |      |      |
| ---- | ------------------- | ---------- | ---------- | ------- | --------- | ---- | ---- |
|      | DenseNet 201 / Bert | No SSE     | No SSE     | 69      | 90+       |      |      |
|      | DenseNet 201 / Bert | 1000, 0.3  | 1000, 0.7  | ~68     | 90+       |      |      |
|      | DenseNet 201 / Bert | 1000, 0.99 | 1000, 0.99 | ~69     | 90+       |      |      |
|      | DenseNet 101 / Bert | No SSE     | No SSE     |         |           |      |      |
|      |                     |            |            |         |           |      |      |
|      |                     |            |            |         |           |      |      |
|      |                     |            |            |         |           |      |      |
|      |                     |            |            |         |           |      |      |
|      |                     |            |            |         |           |      |      |