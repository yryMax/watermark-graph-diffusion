# watermark-graph-diffusion
DMLS 2024 summer intern


TOPIC: adapt watermark method(GS) to discrete graph diffusion(models on social networks, etc.)


# Resources
watermark source hub: https://github.com/ThuCCSLab/Awesome-LM-SSP

graph diffusion source hub: https://github.com/yuanqidu/awesome-graph-generation



# Environment
See https://github.com/cvignac/DiGress
And https://github.com/AlexMRuch/Graph-Sampling/tree/master

# Main Changes
`watermark.py`: watermark embedding & extraction method

`src.diffusion.diffusion_utils.sample_discrete_feature_noise_with_message` this should in replace of the original `sample_discrete_feature_noise` function
to provide the watermark support.

`DiscreteDenoisingDiffusion.sample_batch_simplified` the simplified version of the original `sample_batch` function
which is only about generate synthetic data and get rid of the side effects.



# Benchmark
`python sample.py` to see the synthetic data quality

with watermark: diffusion_model_discrete.py line 617 `z_T = diffusion_utils.sample_discrete_feature_noise_with_message(limit_dist=self.limit_dist, node_mask=node_mask)`

without watermark: diffusion_model_discrete.py line 617 `z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)`

to change the model you need to 
1. change the argpath and modelpath accordingly
2. dump model.node_dist to node_dist.npy
3. copy the second value of marginal distribution to watermark.py line 11

to see the detection quality, run adapt.ipynb
the result is in the block of code with
```angular2html
print("mean error rate", np.mean(errs), "variance error rate", np.var(errs))
print("mean error rate random", np.mean(errs_random), "variance error rate random", np.var(errs_random))
```

# Weights
https://huggingface.co/datasets/Renyi444/watermarking_digress/tree/main
sbm + facebook + flickr

