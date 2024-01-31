from header import *

def getfeature(name='HI', tempname='LyA', wave=1215.6701):

   # Read in the features
   filename = 'atom.npy'
   features = Table(np.load(filename))
   features['wave'] = features['wave']
   features['tempname'] = tempname

   # Get features with this ion                          
   index = np.where(features['name'] == name)
   features = features[index]
   
   # Get the closest match in wavelength
   index = np.argmin(np.absolute(wave - features['wave']))
   feature = features[index]

   name = features['name']
   substring = ""

   for char in str(name[0]):
      if char == 'I' or char == 'V':
         break
      substring += char


   el = element(substring)

   features['mass'] = el.mass

   #easiername
   feature['tempname'] = tempname
   if tempname is None:
      feature['tempname'] = name   
   #if np.abs(feature['wave'] - wave) > 0.2: feature = None
   
   return feature

def genLineTable():
   LyA = getfeature('HI', 'LyA', 1215.670)
   LyB = getfeature('HI', 'LyB', 1025.722)
   LyD = getfeature('HI', 'LyD', 972.5368)
   LyG = getfeature('HI', 'LyG', 949.742)
   LyE = getfeature('HI', 'LyE', 937.8035)
   LyZ = getfeature('HI', 'LyZ', 930.748)
   LyH = getfeature('HI', 'LyH', 926.225)
   LyT = getfeature('HI', 'LyT', 923.150)
   LyI = getfeature('HI', 'LyI', 920.963)
   LyK = getfeature('HI', 'LyK', 919.351)
   LyL = getfeature('HI', 'LyL', 918.351)
   LyM = getfeature('HI', 'LyM', 917.1806)
   LyN = getfeature('HI', 'LyN', 916.429)
   LyX = getfeature('HI', 'LyX', 915.824)
   LyO = getfeature('HI', 'LyO', 915.329)


   OI_1 = getfeature('OI', 'OI_1', 936.63)
   OI_2 = getfeature('OI', 'OI_2', 921.86)

   OII_1 = getfeature('OII', 'OII_1', 834.4655)
   OII_2 = getfeature('OII', 'OII_2', 833.3294)
   OII_3 = getfeature('OII', 'OII_3', 832.7572)
   OIII_1 = getfeature('OIII', 'OIII_1', 832.927)
   OIII_2 = getfeature('OIII', 'OIII_2', 702.332)

   OIV_1 = getfeature('OIV', 'OIV_1', 608.3980)
   OIV_2 = getfeature('OIV', 'OIV_2', 787.711)


   OVI_1 = getfeature('OVI', 'OVI_1', 1031.9261)
   OVI_2 = getfeature('OVI', 'OVI_2', 1037.6167)

   CII_1 = getfeature('CII', 'CII_1', 1334.532)
   CII_2 = getfeature('CII', 'CII_2', 1036.3367)
   CIII = getfeature('CIII', 'CIII', 977.030)
   CIV_1 = getfeature('CIV', 'CIV_1', 1548.187)
   CIV_2 = getfeature('CIV', 'CIV_2', 1550.772)

   SiII_7 = getfeature('SiII', 'SiII_7', 1526.70698)
   SiII_6 = getfeature('SiII', 'SiII_6', 1304.3702)
   SiII_5 = getfeature('SiII', 'SiII_5', 1260.422)
   SiII_4 = getfeature('SiII', 'SiII_4', 1193.2897)
   SiII_3 = getfeature('SiII', 'SiII_3', 1190.4158)
   SiII_2 = getfeature('SiII', 'SiII_2', 1020.6989)
   SiII_1 = getfeature('SiII', 'SiII_1', 989.8731)

   SiIII = getfeature('SiIII', 'SiIII', 1206.500)

   SiIV_1 = getfeature('SiIV', 'SiIV_1', 1393.76018)
   SiIV_2 = getfeature('SiIV', 'SiIV_2', 1402.77291)

   #SiII_4, SiII_1,  SiIII 

   NIII = getfeature('NIII', 'NIII', 989.799)
   NV_1 = getfeature('NV', 'NV_1', 1238.821)
   NV_2 = getfeature('NV', 'NV_2', 1242.804)

   NeVIII_1 = getfeature('NeVIII', 'NeVIII_1', 770.409)
   NeVIII_2 = getfeature('NeVIII', 'NeVIII_2', 780.324)


   SEARCH_LINES = vstack([LyA, LyB, LyD, LyG, LyE, LyZ, LyH, LyT, LyI, LyK, LyL, LyM, LyN, LyX, LyO,
                           OI_1, OI_2, OII_1, OII_2, OII_3, OIII_1, OIII_2, OIV_1, OIV_2, OVI_1, OVI_2, 
                           CII_1, CII_2, CIII, CIV_1, CIV_2, 
                           SiII_1, SiII_2, SiII_3, SiII_4, SiII_5, SiII_6, SiII_7, SiIII, SiIV_1, SiIV_2,
                           NIII, NV_1, NV_2, NeVIII_1, NeVIII_2])
   SEARCH_LINES['wave'] = SEARCH_LINES['wave'].astype(u.Quantity) #* u.Angstrom
   SEARCH_LINES['Gamma'] = SEARCH_LINES['Gamma'].astype(u.Quantity) #* u.Hz
   SEARCH_LINES['Gamma_wavelength'] = SEARCH_LINES['Gamma'].astype(u.Quantity) #* u.Hz

   SEARCH_LINES = QTable(SEARCH_LINES)

   return SEARCH_LINES


SEARCH_LINES = genLineTable()
                           