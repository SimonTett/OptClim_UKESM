# Test can read config file

import os
import UKESMlib
import logging
import pathlib
import StudyConfig

my_logger = logging.getLogger(__file__)
UKESMlib.init_log(my_logger, level='INFO')
if os.environ.get('OPTCLIMTOP', None) is None:
    raise EnvironmentError("Please set the OPTCLIMTOP environment variable to the root of your OptClim installation.")
if os.environ.get('OPT_UKESM_ROOT', None) is None:
    default_path = pathlib.Path(__file__).absolute().parent
    my_logger.warning(f"Please set the OPT_UKESM_ROOT environment variable to the root of your UKESM installation. Using default value = {default_path}.")
    os.environ['OPT_UKESM_ROOT'] = str(default_path)

config = StudyConfig.readConfig(os.environ['OPT_UKESM_ROOT'] + '/configs/dfols4p_UKESM1_1_archer2.json')
# check things work
print("Scaled Targets ============== \n",config.targets(scale=True))
print("Scales ============== \n",config.scales())
print("Start Params ================== \n",config.beginParam())

