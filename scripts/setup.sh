pip install -r requirements.txt
pip install -e .

cd ../deps/rl-toolkit
pip install -e .

cd ../..

mkdir -p data/trained_models
pip install wandb -U