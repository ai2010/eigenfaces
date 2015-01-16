import PIL,os
from PIL import Image
import numpy as np
from scipy import linalg as LA


class PoolImages():
    
      def __init__(self,directory):
         #set of images to compare with the testimage
         self.full_file_paths = self.get_filepaths(directory)#"./photos/training"
         print 'number of training examples:', len(self.full_file_paths)      
         self.images = []
         self.SetImagesFromFolder()
         #determine size each vector
         self.imsize = len(self.images[0])
         #calculate the average
         self.Nimages = len(self.images)
         self.Avimage = []
         self.CalcAverage()
         #calculate diff
         self.diffimages = [0 for i in range(self.Nimages)]
         self.CalcDiffImagesAv()
         #compute the eigenvectors of the C=L^T
         self.uvecs = [[0.0 for j in range(self.imsize)] for i in range(self.Nimages)]
         self.CalcEigenvectors()
         #calc images projections:
         self.projdiffimages = [[0 for j in range(self.Nimages)] for i in range(self.Nimages)]
         self.CalcProjectionImages()
         
      def get_filepaths(self,directory):
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
          
      def SetImagesFromFolder(self):
          for f in self.full_file_paths:
            im = Image.open(f).convert('L')
            pix = im.load()
            w=im.size[0]
            h=im.size[1]
            imagetmp = []
            for i in range(w):
              for j in range(h):
                imagetmp.append(pix[i,j]) 
            self.images.append(imagetmp)
      
      def CalcAverage(self):
          for i in range(self.imsize):
            tmpvalue = sum(row[i] for row in self.images)
            self.Avimage.append(tmpvalue/self.Nimages)
      
      def DumpAverage(self):
          #see the av output
          img=Image.open(self.full_file_paths[0]).convert('L')# grayscale
          pix=img.load()
          k=0
          for i in range(img.size[0]):
            for j in range(img.size[1]):
              pix[i,j] = self.Avimage[k]
              k=k+1
          img.save("average.jpg")
          
      def CalcDiffImagesAv(self):
          for i in range(self.Nimages):
            tmpdiff = []
            for j in range(self.imsize):
              tmpdiff.append(float(-self.Avimage[j]+self.images[i][j]))
          #diffimages.append(tmpdiff)
            self.diffimages[i] = tmpdiff
            
      def CalcEigenvectors(self):
          #calculate transponse covariance matrix
          L = [[0 for j in range(self.Nimages)] for i in range(self.Nimages)]
          for i in range(self.Nimages):
            for j in range(self.Nimages):
                L[i][j] = np.dot(self.diffimages[i],self.diffimages[j])

          #calc eigenvalues/vectors
          evals, evecs = LA.eig(L)

          for i in range(self.Nimages):
              for j in range(self.Nimages):
                  self.uvecs[i] = map(lambda x,y:x+evecs[i][j]*y, self.uvecs[i],self.diffimages[j])
           
      def CalcProjectionImages(self):
          for i in range(self.Nimages):
            for j in range(self.Nimages):
              self.projdiffimages[i][j] = np.dot(self.uvecs[j],self.diffimages[i])
      



class CheckImage():
    
      def __init__(self,FILENAME,directory):
          self.imagetest = []
          self.GetImage(FILENAME)
          self.poolims= PoolImages(directory)
          #calc diff and projection into subspace
          self.utest = [0 for i in range(self.poolims.Nimages)]
          self.uvecs = self.poolims.uvecs
          self.Avim = self.poolims.Avimage
          self.CalcProjectionIm()
          self.projectionims = self.poolims.projdiffimages
          #self.DeterminePoolImage
          
      def GetImage(self,filename):
          imtest=Image.open(filename).convert('L')# grayscale
          pixtest = imtest.load()
          w=imtest.size[0]
          h=imtest.size[1]
          for i in range(w):
            for j in range(h):
              self.imagetest.append(pixtest[i,j])

      def CalcProjectionIm(self):
          for i in range(self.poolims.Nimages):
              self.utest[i] = sum(map(lambda x,y,z: x*(y-z),self.uvecs[i],self.imagetest,self.Avim))

      def DeterminePoolImage(self):
          #calc distances:
          mindist = 1000
          indx=0
          for i in range(self.poolims.Nimages):
            if i==0:
               mindist= self.dist(self.utest,self.projectionims[i])
            else:
              if self.dist(self.utest,self.projectionims[i])<mindist:
                mindist = self.dist(self.utest,self.projectionims[i])
                indx =i
          print 'index=',indx,'mindist=',mindist
          #show closer image
          imgresult=Image.open(self.poolims.full_file_paths[indx]).convert('L')
          imgresult.save("result.jpg")    
                
      def dist(self,v1,v2):
          '''
          calculate the distance between 2 vectors
          '''
          d=0
          for x in range(len(v1)):
            d = d+(v1[x]-v2[x])*(v1[x]-v2[x]) 
          return d    
          

#pool images
folder = "./photos/training"

#test image here:
FILENAME='testimage.jpg' #image can be in gif jpeg or png format
#run script
test = CheckImage(FILENAME,folder)
test.DeterminePoolImage()



  
