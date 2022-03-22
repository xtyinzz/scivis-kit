import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--nncfg',
    help='yaml config file for the network module (training, logs, etc.)',
    default='nn.yaml'
  )
  parser.add_argument(
    '--resume',
    help='resume training or start over',
    action='store_true'
  )
  parser.add_argument(
    '--testcfg',
    help='yaml config file for the visualization (network path, data paths, .etc.)',
    default='nntest.yaml'
  )
  args = parser.parse_args()
  return args