量化命令行

```
lmdeploy lite auto_awq    $HF_MODEL   --calib-dataset 'wikitext2'   --calib-samples 128   --calib
-seqlen 2048   --w-bits 4   --w-group-size 128   --batch-size 1   --work-dir $WORK_DIR
```


量化过程
```
lmdeploy lite auto_awq    $HF_MODEL   --calib-dataset 'wikitext2'   --calib-samples 128   --calib-seqlen 2048   --w-bits 4   --w-group-size 128   --batch-size 1   --work-dir $WORK_DIR
Warning: we cast model to float16 to prevent OOM. You may enforce it bfloat16 by `--dtype bfloat16`
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:25<00:00, 12.63s/it]
Move model.embed_tokens to GPU.
Move model.layers.0 to CPU.
Move model.layers.1 to CPU.
Move model.layers.2 to CPU.
model.layers.0.mlp.up_proj weight packed.
model.layers.0.mlp.down_proj weight packed.
model.layers.1.self_attn.q_proj weight packed.
model.layers.1.self_attn.k_proj weight packed.
```

查看量化后模型

```
(xtuner) root@intern-studio-031316:~/lmdeploy# cd /root/internlm3-8b-instruct-4bit/
(xtuner) root@intern-studio-031316:~/internlm3-8b-instruct-4bit# ls
config.json                 inputs_stats.pth                  model.safetensors.index.json  tokenization_internlm3.py
configuration_internlm3.py  model-00001-of-00002.safetensors  outputs_stats.pth             tokenizer.model
generation_config.json      model-00002-of-00002.safetensors  special_tokens_map.json       tokenizer_config.json
```