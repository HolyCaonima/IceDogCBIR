

class ImageReader:
    
    __ImageClass=''
    __ParsedUrl=[]
    __DataUrl='http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='
    __CurrentPtr=0;
   
    def __init__(self,imageClass,url):
        self.__DataUrl=self.__DataUrl+url
        self.__ImageClass=imageClass
        # begin get reqlist
        import urllib.request
        res_data=urllib.request.urlopen(self.__DataUrl)
        noParsedUrl=res_data.read().decode('utf-8')
        # begin parsing
        self.__ParsedUrl=noParsedUrl.split('\r\n')
        self.__ParsedUrl.pop()
        
    def getOneUrl(self):
        if self.__CurrentPtr==len(self.__ParsedUrl):
            return 0
        tempUrl=self.__ParsedUrl[self.__CurrentPtr]
        self.__CurrentPtr+=1
        return tempUrl
        
    def getBatch(self,batchSize):
        import numpy
        import urllib.request
        result=[]
        # for batch get image
        while batchSize>0:
            try:
                tempUrl=self.getOneUrl()
                imgType=tempUrl.split('.')
                imgType=imgType[len(imgType)-1]
                print('try: '+tempUrl)
                # connect and read the data
                cnt=urllib.request.urlopen(tempUrl,timeout=0.5)
                dat=cnt.read()
                # to fileStream and put to image
                import io
                from PIL import Image
                file=io.BytesIO(dat)
                img=Image.open(file)
                img=img.resize((100,100))
                img.show()
                result.append(numpy.array(img))
                print('getOne : '+imgType+"  :  ("+str(result[len(result)-1].shape[0])+','+str(result[len(result)-1].shape[1])+')')
                batchSize=batchSize-1
            except Exception as e:
                print('failOne')
        return numpy.array(result)
            
    def imageCount(self):
        return len(self.__ParsedUrl)
    
    def imageClass(self):
        return self.__ImageClass;
        