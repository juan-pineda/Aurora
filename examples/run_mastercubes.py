from aurora import main
from sys import argv

if __name__ == '__main__':
	#ConfigFile = argv[1]
	ConfigFile = 'params_20_0.config'

	main.spectrom_mock(ConfigFile)
