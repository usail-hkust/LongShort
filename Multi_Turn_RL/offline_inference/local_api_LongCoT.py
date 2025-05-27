import uvicorn
from multiprocessing import Process
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-host", type=str, default='0.0.0.0')
    parser.add_argument("-port", type=str, default=8876)
    parser.add_argument("-workers", type=int, default=1) # multi threads
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    uvicorn.run(app='local_app_LongCoT:app', host=args.host, port=args.port, workers=args.workers)

