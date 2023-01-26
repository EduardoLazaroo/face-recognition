from abc import ABCMeta, abstractmethod


class ListClass:

    def __init__(self):
        pass

    def generate_list(self, quantity):
        final = []
        for i in range(quantity):
            final.append([])
        return final


class printExecute(metaclass=ABCMeta):
    @abstractmethod
    def printMessage(self):
        pass


class checkCam(printExecute):
    def printMessage(self):
        print("Executando camera")


class checkFace(printExecute):
    def printMessage(self):
        print("Face reconhecida")


class ADM:
    def printADM(self, printexecute):
        printexecute.printMessage()


class errorCam:
    def printError(self):
        print("Erro na camera")


class errorCamAdapter(printExecute):
    def __init__(self, errorCam):
        self.__errorCam = errorCam

    def printMessage(self):
        self.__errorCam.printError()
