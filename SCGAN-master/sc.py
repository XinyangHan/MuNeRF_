from options.test_options import TestOptions
from models.models import create_model
from SCDataset.SCDataset import SCDataLoader
import pdb

opt = TestOptions().parse()
# pdb.set_trace()
data_loader = SCDataLoader(opt)
SCGan = create_model(opt, data_loader)
if opt.phase=='train':
    SCGan.train()
elif opt.phase=='test':
    SCGan.test()
print("Finished!!!")




