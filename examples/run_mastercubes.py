from src import main as au
from sys import argv

if __name__ == '__main__':
	#ConfigFile = argv[1]
	ConfigFile = 'params_20_0.config'

	au.spectrom_mock(ConfigFile)
