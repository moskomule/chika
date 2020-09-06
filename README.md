# chika

`chika` is a simple and easy config tool for hierarchical configurations.

## Requirements

* Python>=3.7

## Usage

Write typed configurations using `chika.config`, which is similar to `dataclass`.

```python
# main.py
import chika

@chika.config
class ModelConfig:
    name: str = chika.choices('resnet', 'densenet')
    depth: int = 10

@chika.config
class DataConfig:
    # values that needs to be specified
    name: str = chika.required()

@chika.config
class OptimConfig:
    # 
    steps: List[int] = chika.sequence(100, 150)

@chika.config
class Config:
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig

    seed: int = chika.with_help(1, help="random seed")
    use_amp: bool = False
    gpu: int = chika.choices(range(torch.cuda.device_count()), help="id of gpu")


@chika.main(Config)
def main(cfg: Config):
    model = ModelRegistry(cfg.model)
    ...

```

```yaml
# config/densenet.yaml
model:
  name: densenet
  depth: 12 
```

### Expected Behavior

```commandline
python main.py --use_amp
# cfg.use_amp == True

python main.py --model config/densenet.yaml
# cfg.model.name == densenet
# cfg.model.depth == 12

python main.py --model config/densenet.yaml --model.depth 24
# cfg.model.name == densenet
# cfg.model.depth == 24

python main.py --optim.decay_steps 120 150
# config.optim.decay_steps == [120, 150]
```

### Other APIs

```python
from chika import ChikaConfig
cfg = ChikaConfig.from_dict(...)

cfg.to_dict()
# {"model": {"name": "resnet", "zero_init": True, ...}, ...}
```

### Known issues

-[ ] config cannot be nested twice. `Config(Config(...))` is valid, but `Config(Config(Config(...)))` is invalid.