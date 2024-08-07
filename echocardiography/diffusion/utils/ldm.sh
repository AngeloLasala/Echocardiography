## define the entire process of LDM. Train, sample the validation data and evaluate the performance of the model
## in thid file, we have to define only the .yaml file and the trial number

## 1. train the model
python -m echocardiography.diffusion.tools.train_ldm --data eco --trial trial_7

## 2. sample the fake image
python -m echocardiography.diffusion.tools.sample_ldm --data eco --trial trial_7 --experiment ldm_1 --epoch 20
python -m echocardiography.diffusion.tools.sample_ldm --data eco --trial trial_7 --experiment ldm_1 --epoch 40
python -m echocardiography.diffusion.tools.sample_ldm --data eco --trial trial_7 --experiment ldm_1 --epoch 60
python -m echocardiography.diffusion.tools.sample_ldm --data eco --trial trial_7 --experiment ldm_1 --epoch 80
python -m echocardiography.diffusion.tools.sample_ldm --data eco --trial trial_7 --experiment ldm_1 --epoch 100

## 3. evaluate the performance of the model
python -m echocardiography.diffusion.evaluation.fid --trial trial_7 --experiment ldm_1


