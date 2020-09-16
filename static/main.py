import sys
import subprocess
from subprocess import call

timestamp = sys.argv[-2]
file_ext = sys.argv[-1]

def main():
    # print "ha ha ha ha ha ah ah ah aha ha ah ah aha ha ah aha ha ah aha ha ha"
    call(['python', 'static/box.py', timestamp, file_ext])
    # call(['python', 'static/preprocessing_and_segmentation.py', timestamp, file_ext])
    # print "ha ha ha ha ha ah ah ah aha ha ah ah aha ha ah aha ha ah aha ha ha"
    call(['python', 'static/api.py', timestamp, file_ext])
    # print "ha ha ha ha ha ah ah ah aha ha ah ah aha ha ah aha ha ah aha ha ha"
    call(['python', 'static/error_correction_and_output.py', timestamp, file_ext])
    # print "ha ha ha ha ha ah ah ah aha ha ah ah aha ha ah aha ha ah aha ha ha"

if __name__ == '__main__':
    main()