from options import Options
from trainer import TrainManager
from inference import InferenceManager


OPTIONS = Options()
OPTIONS = OPTIONS.parse()
OPTIONS.mode = 'train'
OPTIONS.training_datasets = ['mscoco']
OPTIONS.log_path = '/home/patel.aryam/stereo-from-mono/out'
OPTIONS.disable_synthetic_augmentation = True


if OPTIONS.mode == 'train':
    print('In training mode!')
    TRAINER = TrainManager(OPTIONS)
    TRAINER.train()

elif OPTIONS.mode == 'inference':
    print('In inference mode!')
    TESTER = InferenceManager(OPTIONS)
    TESTER.run_inference()

else:
    raise NotImplementedError
