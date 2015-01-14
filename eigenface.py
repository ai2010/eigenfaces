import PIL,os
from PIL import Image
import numpy as np
from scipy import linalg as LA

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


def dist(v1,v2):
    '''
    calculate the distance between 2 vectors
    '''
    d=0
    for x in range(len(v1)):
      d = d+(v1[x]-v2[x])*(v1[x]-v2[x]) 
    return d

#set of images to compare with the testimage
full_file_paths = get_filepaths("./photos/training")
print 'number of training examples:', len(full_file_paths)

images = []
for f in full_file_paths:
  im = Image.open(f).convert('L')
  pix = im.load()
  w=im.size[0]
  h=im.size[1]
  imagetmp = []
  for i in range(w):
    for j in range(h):
      imagetmp.append(pix[i,j]) 
  images.append(imagetmp)

#determine size each vector
imsize = len(images[0])
#calculate the average
Nimages = len(images)
Avimage = []

for i in range(imsize):
  tmpvalue = sum(row[i] for row in images)
  Avimage.append(tmpvalue/Nimages)


#see the av output
img=Image.open(full_file_paths[0]).convert('L')# grayscale
pix=img.load()
k=0
for i in range(img.size[0]):
  for j in range(img.size[1]):
    pix[i,j] = Avimage[k]
    k=k+1
img.save("average.jpg")

#calculate diff
diffimages = [0 for i in range(Nimages)]
for i in range(Nimages):
  tmpdiff = []
  for j in range(imsize):
    tmpdiff.append(float(-Avimage[j]+images[i][j]))
#diffimages.append(tmpdiff)
  diffimages[i] = tmpdiff


#calculate transponse covariance matrix
L = [[0 for j in range(Nimages)] for i in range(Nimages)]

for i in range(Nimages):
  for j in range(Nimages):
      #tmpL=0
      #for k in range(imsize):
           #tmpL = tmpL + diffimages[i][k]*diffimages[j][k]
      L[i][j] = np.dot(diffimages[i],diffimages[j])

#calc eigenvalues/vectors
evals, evecs = LA.eig(L)

#compute the eigenvectors of the C=L^T
uvecs = [[0.0 for j in range(imsize)] for i in range(Nimages)]

for i in range(Nimages):
    for j in range(Nimages):
        uvecs[i] = map(lambda x,y:x+evecs[i][j]*y, uvecs[i],diffimages[j])
#calc images projections:
projdiffimages = [[0 for j in range(Nimages)] for i in range(Nimages)]
for i in range(Nimages):
  for j in range(Nimages):
    projdiffimages[i][j] = np.dot(uvecs[j],diffimages[i])

#test image here:
FILENAME='testimage.jpg' #image can be in gif jpeg or png format

imtest=Image.open(FILENAME).convert('L')# grayscale
pixtest = imtest.load()
w=imtest.size[0]
h=imtest.size[1]
imagetest = []
for i in range(w):
  for j in range(h):
    imagetest.append(pixtest[i,j]) 
#calc diff and projection into subspace
utest = [0 for i in range(Nimages)]

for i in range(Nimages):
    utest[i] = sum(map(lambda x,y,z: x*(y-z),uvecs[i],imagetest,Avimage))
      

#calc distances:
mindist = 1000
indx=0
for i in range(Nimages):
  if i==0:
     #print len(utest), len(projdiffimages[0])
     mindist= dist(utest,projdiffimages[i])
     #print mindist
  else:
    if dist(utest,projdiffimages[i])<mindist:
      mindist = dist(utest,projdiffimages[i])
      indx =i


print 'index=',indx,'mindist=',mindist
#show closer image
imgresult=Image.open(full_file_paths[indx]).convert('L')
imgresult.save("result.jpg")
  
