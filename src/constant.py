import os

# Original files
PROPERTIES_2016 = os.path.expandvars("$ZILLOW/data/properties_2016.csv")
TRAIN_2016 = os.path.expandvars("$ZILLOW/data/train_2016_v2.csv")
SAMPLE_SUBMIT = os.path.expandvars("$ZILLOW/data/sample_submission.csv")

# Pickle files
PROPERTIES_2016_PKL = os.path.expandvars("$ZILLOW/data/properties_2016.pkl")
TRAIN_2016_PKL = os.path.expandvars("$ZILLOW/data/train_2016_v2.pkl")

# Feature Factory
FEATURE_FACTORY_DEFINITIONS = os.path.expandvars("$ZILLOW/features/feature_definitions.txt")
ORACLE_FACTORY_DEFINITIONS = os.path.expandvars("$ZILLOW/features/oracle_definitions.txt")
FEATURE_FACTORY_TRAIN = os.path.expandvars("$ZILLOW/features/train/")
FEATURE_FACTORY_TEST_1610 = os.path.expandvars("$ZILLOW/features/test/1610")
FEATURE_FACTORY_TEST_1611 = os.path.expandvars("$ZILLOW/features/test/1611")
FEATURE_FACTORY_TEST_1612 = os.path.expandvars("$ZILLOW/features/test/1612")
FEATURE_FACTORY_TEST_1710 = os.path.expandvars("$ZILLOW/features/test/1710")
FEATURE_FACTORY_TEST_1711 = os.path.expandvars("$ZILLOW/features/test/1711")
FEATURE_FACTORY_TEST_1712 = os.path.expandvars("$ZILLOW/features/test/1712")