# ngym_usage

## Setup

`git clone --recursive https://github.com/neurogym/ngym_usage.git`

`cd ngym_usage`

`pip install -r requirements.txt`

## Note

Code is currently incompatible with the latest OpenAI/gym version (version 0.21.0).
To make the code compatible with gym:
* either change line 238 in `neurogym/envs/registration.py` from `register(id=env_id, entry_point=entry_point)` to `register(id=env_id, entry_point=entry_point, order_enforce=False)`
* or downgrade gym version
