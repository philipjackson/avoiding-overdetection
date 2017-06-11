import pickle, glob, os

filenames = glob.glob("results/*")
if not filenames:
    raise IOError( "No params file found, exiting")
filename = max(filenames, key=os.path.getctime) # choose most recnt matching file
fin = open(filename, 'rb')
log = pickle.load(fin)
fin.close()
