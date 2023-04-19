import logging
import os
import numpy as np
import cv2
import scipy.linalg as la
from copy import copy
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import math
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCamera,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer, vtkWindowToImageFilter,
)

import unittest
import pathlib
import scipy
import scipy.io as sio

#
# CTRegistrationHandEye
#

class CTRegistrationHandEye(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CTRegistrationHandEye"  # TODO: make this more human readable by adding spaces
        self.parent.categories = [
            "Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#CTRegistrationHandEye">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
  Add data sets to Sample Data module.
  """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # CTRegistrationHandEye1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='CTRegistrationHandEye',
        sampleName='CTRegistrationHandEye1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'CTRegistrationHandEye1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='CTRegistrationHandEye1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='CTRegistrationHandEye1'
    )

    # CTRegistrationHandEye2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='CTRegistrationHandEye',
        sampleName='CTRegistrationHandEye2',
        thumbnailFileName=os.path.join(iconsPath, 'CTRegistrationHandEye2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='CTRegistrationHandEye2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='CTRegistrationHandEye2'
    )


#
# CTRegistrationHandEyeWidget
#

class CTRegistrationHandEyeWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent=None):
        """
    Called when the user opens the module the first time and the widget is initialized.
    """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        self.saveFolder = pathlib.Path(__file__).parent.resolve()


    def setup(self):
        """
    Called when the user opens the module the first time and the widget is initialized.
    """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/CTRegistrationHandEye.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = CTRegistrationHandEyeLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.imageThresholdSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        self.ui.invertOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        ##########################
        self.ui.pushButton.connect('clicked(bool)', self.Start)

        ##########################

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
    Called when the application closes and the module widget is destroyed.
    """
        self.removeObservers()

    def enter(self):
        """
    Called each time the user opens this module.
    """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
    Called each time the user opens a different module.
    """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
    Called just before the scene is closed.
    """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
    Called just after the scene is closed.
    """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
    Ensure parameter node exists and observed.
    """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
        self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
        self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
        self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
            self.ui.applyButton.toolTip = "Compute output volume"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input and output volume nodes"
            self.ui.applyButton.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        """
    Run processing when user clicks "Apply" button.
    """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
                                   self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked,
                                   showResult=False)

    ###############################
    def Start(self):
        self.logic.logicStart()


###############################

#
# CTRegistrationHandEyeLogic
#

class CTRegistrationHandEyeLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self):
      """
      Called when the logic class is instantiated. Can be used for initializing member variables.
        """
      ScriptedLoadableModuleLogic.__init__(self)
      self.saveFolder = pathlib.Path(__file__).parent.resolve()


    def setDefaultParameters(self, parameterNode):
        """
    Initialize parameter node with default settings.
    """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be thresholded
    :param outputVolume: thresholding result
    :param imageThreshold: values above/below this threshold will be set to 0
    :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
    :param showResult: show output volume in slice viewers
    """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True,
                                 update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime - startTime:.2f} seconds')

    ###############################################

    def PixelValidation(self, T, X, Q, A):
        """
        INPUTS:  T : (4x4) hand eye calibration matrix
                 X : (3xn) 3D coordinates , ( tracker space)
                 Q : (2xn) 2D pixel locations (image space)
                 A : (3x3) camera matrix
        OUTPUT:  pixelErrors : Column vector of pixel errors
        """
        pixels = []
        pixelErrors = []

        for k in range(X.shape[1]):
            point = X[:, k]
            point = np.reshape(point, (3, 1))

            pix = Q[:, k]
            pix = np.reshape(pix, (2, 1))

            point = np.vstack((point, 1))

            # Register 3D point to line
            cameraPoint = T @ point

            # Convert 3d point to homogeneous coordinates
            cameraPoint = cameraPoint / cameraPoint[2]
            cameraPoint = cameraPoint[0:2, :]
            cameraPoint = np.vstack((cameraPoint, 1))

            # Project point onto image using camera intrinsics
            pixel = A @ cameraPoint
            pixels.append(pixel)

            xError = abs(pixel[0, 0] - pix[0, 0])
            yError = abs(pixel[1, 0] - pix[1, 0])

            pixelErrors.append(np.sqrt(xError * xError + yError * yError))

        pixelErrors = np.reshape(pixelErrors, (X.shape[1], 1))
        return pixels, pixelErrors

    def trans_to_matrix(self, trans):
        """ Convert a numpy.ndarray to a vtk.vtkMatrix4x4 """
        matrix = vtk.vtkMatrix4x4()
        for i in range(trans.shape[0]):
            for j in range(trans.shape[1]):
                matrix.SetElement(i, j, trans[i, j])
        return matrix

    def main(self, M_int_est, M_ext_est, P_2D, P_3D):

        h = 1080
        w = 1349

        # Create a transform node in you declarations
        ExtrinsicTransformNode = None
        # try:
        #     ExtrinsicTransformNode = slicer.mrmlScene.GetFirstNode("extrinsic")
        #     # Clear the list
        #     slicer.mrmlScene.RemoveNode(ExtrinsicTransformNode)
        # except:
        #     self.ExtrinsicTransformNode = slicer.vtkMRMLLinearTransformNode()  # CREATING THE TRANSFORM NODE
        #     slicer.mrmlScene.AddNode(self.ExtrinsicTransformNode)  # <--- ADDING IT TO THE SCENE
        #     self.ExtrinsicTransformNode.SetName(
        #         'extrinsic')  # <--- THE NAME IS GOING TO APPEAR IN THE TRANSFORM AND DATA MODULES (NAME OF YOUR NODE)
        #
        # #
        # # idk why the creating the node code has to be repeated from the except section, but for some reason it works this way so there will only ever be one extrinsic transform node
        # self.ExtrinsicTransformNode = slicer.vtkMRMLLinearTransformNode()  # CREATING THE TRANSFORM NODE
        # slicer.mrmlScene.AddNode(self.ExtrinsicTransformNode)  # <--- ADDING IT TO THE SCENE
        # self.ExtrinsicTransformNode.SetName(
        #     'extrinsic')  # <--- THE NAME IS GOING TO APPEAR IN THE TRANSFORM AND DATA MODULES (NAME OF YOUR NODE)

        intrinsic = M_int_est

        extrinsic = M_ext_est

        # # what is the calibration matrix
        # self.extrinsicObj = vtk.vtkMatrix4x4()
        # self.extrinsicObj.DeepCopy((M_ext_est[0, 0], M_ext_est[0, 1], M_ext_est[0, 2], M_ext_est[0, 3],
        #                             M_ext_est[1, 0], M_ext_est[1, 1], M_ext_est[1, 2], M_ext_est[1, 3],
        #                             M_ext_est[2, 0], M_ext_est[2, 1], M_ext_est[2, 2], M_ext_est[2, 3],
        #                             0, 0, 0, 1))



        # create camera

        layoutManager = slicer.app.layoutManager()
        view = layoutManager.threeDWidget(0).threeDView()
        threeDViewNode = view.mrmlViewNode()
        cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(threeDViewNode)
        renderWindow = view.renderWindow()
        renderers = renderWindow.GetRenderers()
        renderer = renderers.GetItemAsObject(0)
        camera = cameraNode.GetCamera()


        viewLogic = slicer.app.applicationLogic().GetViewLogic(threeDViewNode)

        reader = vtk.vtkPNGReader()
        reader.SetFileName("C:/d/CTRegistrationHandEye/CTRegistrationHandEye/brain.png")
        reader.Update()


        #
        # intrinsics
        #

        viewLogic.StartCameraNodeInteraction(cameraNode.LookFromAxis)

        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        f = intrinsic[0, 0]

        # print("f = ", f)

        # convert the principal point to window center (normalized coordinate system) and set it
        wcx = -2 * (cx - float(w) / 2) / w
        wcy = 2 * (cy - float(h) / 2) / h
        camera.SetWindowCenter(wcx, wcy)

        # convert the focal length to view angle and set it
        view_angle = 180 / math.pi * (2.0 * math.atan2(h / 2.0, f))
        camera.SetViewAngle(view_angle)

        #
        # extrinsics
        #

        # apply the transform to scene objects
        camera.SetModelTransformMatrix(self.trans_to_matrix(extrinsic))

        # the camera can stay at the origin because we are transforming the scene objects
        camera.SetPosition(0, 0, 0)

        # look in the +Z direction of the camera coordinate system
        camera.SetFocalPoint(0, 0, f)

        # the camera Y axis points down
        camera.SetViewUp(0, -1, 0)

        renderer.ResetCameraClippingRange()
        viewLogic.EndCameraNodeInteraction()

        # set background to image
        texture = vtk.vtkTexture()
        texture.SetInputConnection(reader.GetOutputPort())
        renderer.SetTexturedBackground(True)
        renderer.SetBackgroundTexture(texture)


        # # move model (does same thing as apply the transform to scene objects)
        # modelNode = slicer.util.getNode("Model_2_2")
        #
        # # assign the model to follow a transformNode
        # modelNode.SetAndObserveTransformNodeID(self.ExtrinsicTransformNode.GetID())
        #
        # #The transform node tell him what matrix(calibration)
        # ExtrinsicTransformNode.SetMatrixTransformToParent(self.extrinsicObj)




        # Draw 3D points (test)
        # Create markup nodes to show 3D tracked data, make sure there is not already one created

        P_3D_ExtX = []
        P_3D_ExtY = []
        P_3D_ExtZ = []

        #multiply 3D points by extrinsic matrix to visualize
        for k in range(P_3D.shape[1]):
            point = P_3D[:, k]
            point = np.reshape(point, (3, 1))

            point = np.vstack((point, 1))

            # Register 3D point to line
            cameraPoint =  point #extrinsic @ point * -1

            P_3D_ExtX.append(cameraPoint[0][0])
            P_3D_ExtY.append(cameraPoint[1][0])
            P_3D_ExtZ.append(cameraPoint[2][0])

        markupsNode_3D = None
        try:
            markupsNode_3D = slicer.util.getNode("3DFiducials")
            # Clear the list
            markupsNode_3D.RemoveAllControlPoints()
        except:
            markupsNode_3D = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
            markupsNode_3D.SetName("3DFiducials")

        for i in range(np.shape(P_3D)[1]):
            # draw fiducials following spatial tracking for visual validation
            markupsNode_3D.AddControlPoint(P_3D_ExtX[i], P_3D_ExtY[i], P_3D_ExtZ[i])


    def cameraCombinedCalibration2(self, P_2D, P_3D):
        """
     Performed a combine camera calibration (both the intrinsic and extrinsic
     (i.e. hand-eye) based on corresponding 2D pixels and 3D points.

     Inputs:  P_2D - 2xN pixel coordinates
              P_3D - 3xN 3D coordinates, it is assumed that each 3D points is
              projected to the image plane, thus there is a known
              correspondence between P_2D and P_3D

     Outputs: M_int_est, estimated camera intrinsic parameters:
                       = [ fx 0 cx
                           0 fy cy
                           0  0  1 ]
              M_ext_est, estimate camera extrinsic parameters (i.e. hand-eye)
                       = [ R_3x3, t_3x1
                               0  1 ]
    """

        # space allocation for outputs
        M_int_est = np.identity(3)
        M_ext_est = np.identity(4)

        # size of the input
        N = P_2D.shape[1]
        if P_2D.shape[1] == P_3D.shape[1]:

            # construct the system of linear equations
            A = np.empty((0, 12))
            for i in range(N):
                a = np.array([[P_3D[0, i], P_3D[1, i], P_3D[2, i], 1, 0, 0, 0, 0, -P_2D[0, i] * P_3D[0, i],
                               -P_2D[0, i] * P_3D[1, i], -P_2D[0, i] * P_3D[2, i], -P_2D[0, i]]])
                b = np.array([[0, 0, 0, 0, P_3D[0, i], P_3D[1, i], P_3D[2, i], 1, -P_2D[1, i] * P_3D[0, i],
                               -P_2D[1, i] * P_3D[1, i], -P_2D[1, i] * P_3D[2, i], -P_2D[1, i]]])

                c = np.vstack((a, b))
                A = np.vstack((A, c))

            # The answer is the eigen vector corresponding to the single zero eivenvalue of the matrix (A' * A)
            D, V = la.eig(A.conj().T @ A)
            min_idx = np.where(D == min(D))[0][0]
            m = V[:, min_idx]

            # note that m is arranged as:
            # m = [ m11 m12 m13 m14 m21 m22 m23 m24 m31 m32 m33 m34]
            # rearrange to form:
            # m = [ m11 m12 m13 m14 ;
            #       m21 m22 m23 m24 ;
            #       m31 m32 m33 m34 ];

            m = np.reshape(m, (3, 4))

            # m is known as the projection matrix, basically it is the intrinsic matrix multiplied with the extrinsic (hand-eye calibration) matrix
            # The first step to resolve intrinsic/extrinsic matrices from m is to find the scaling factor. Note that the last row [m31 m32 m33] is the last row of the rotation matrix R, thus one can find the scale there
            gamma = np.absolute(np.linalg.norm(m[2, 0:3]))

            # determining the translation in Z and the sign of sigma
            gamma_sign = np.sign(m[2, 3])

            # due to the way we construct our viewing axis, we know that the objects must be IN FRONT of the camera, thus the translation must always be POSITIVE in the Z direction
            M = gamma_sign / gamma * m
            M_proj = M

            # translation in z
            Tz = M[2, 3]

            # third row of the rotation matrix
            M_ext_est[2, :] = M[2, :]

            # principal points
            ox = np.dot(M[0, 0:3], M[2, 0:3])
            oy = np.dot(M[1, 0:3], M[2, 0:3])

            # focal points
            fx = np.sqrt(np.dot(M[0, 0:3], M[0, 0:3]) - ox * ox)
            fy = np.sqrt(np.dot(M[1, 0:3], M[1, 0:3]) - oy * oy)

            # construct the output
            M_int_est[0, 2] = ox
            M_int_est[1, 2] = oy
            M_int_est[0, 0] = fx
            M_int_est[1, 1] = fy

            # 1st row of the rotation matrix
            M_ext_est[0, 0:3] = gamma_sign / fx * (ox * M[2, 0:3] - M[0, 0:3])

            # 2nd row of the rotation matrix
            M_ext_est[1, 0:3] = gamma_sign / fy * (oy * M[2, 0:3] - M[1, 0:3])

            # translation in x
            M_ext_est[0, 3] = gamma_sign / fx * (ox * Tz - M[0, 3])

            # translation in y
            M_ext_est[1, 3] = gamma_sign / fy * (oy * Tz - M[1, 3])

            M_ext_est_neg = copy(M_ext_est)
            M_ext_est_neg[0, :] = -1 * M_ext_est_neg[0, :]
            M_ext_est_neg[1, :] = -1 * M_ext_est_neg[1, :]

            if (np.linalg.norm(M_int_est @ M_ext_est[0:3, :] - M_proj, 'fro')) > (
                    np.linalg.norm(M_int_est @ M_ext_est_neg[0:3, :] - M_proj, 'fro')):
                M_ext_est = copy(M_ext_est_neg)

            # given the 3x4 projection matrix M, calculate the projection error between the paired 2D/3D fiducials
            m, n = np.shape(P_3D)

            # calculate the projection of P3D given M
            temp = M @ np.vstack((P_3D, np.ones((1, n))))
            P = np.zeros((2, n))
            P[0, :] = temp[0, :] / temp[2, :]
            P[1, :] = temp[1, :] / temp[2, :]

            # mean projection error, i.e. euclidean distance between P3D after projection from P2D
            p = P - P_2D
            fre = []
            for i in range(n):
                x = np.linalg.norm(p[:, i])
                fre.append(x)

            fre = np.sum(fre) / n

        else:
            print("error: 2D and 3D matricies of different lengths")

        return M_int_est, M_ext_est, M_proj, fre

    def logicStart(self):

        # load in CT fiducials volume and extract locations of fiducials (points w/ nonzero values)
        CT_idx = slicer.util.getNode('CT_idx_small')
        CT_idx = slicer.util.arrayFromVolume(CT_idx)
        points_3D = np.nonzero(CT_idx)

        # sort 3D fiducial locations into xyz
        pZ_3D = points_3D[0].tolist()
        pY_3D = points_3D[1].tolist()
        pX_3D = points_3D[2].tolist()

        # make list of 3D fiducial indices to use for extracting 2D fiducial locations
        indices = []

        for i in range(len(pX_3D)):
            indices.append(CT_idx[pZ_3D[i], pY_3D[i], pX_3D[i]])


        # check for duplicates and remove in both indices list and 3D fiducials lists
        def list_duplicates(seq):
            seen = set()
            seen_add = seen.add
            # adds all elements it doesn't know yet to seen and all other to seen_twice
            seen_twice = set(x for x in seq if x in seen or seen_add(x))
            # turn the set into a list (as requested)
            return list(seen_twice)

        duplicates = list_duplicates(indices)

        # print(len(indices))

        for i in range(len(duplicates)):
            while True:
                try:
                    a = indices.index(duplicates[i])
                    indices.pop(a)
                    pX_3D.pop(a)
                    pY_3D.pop(a)
                    pZ_3D.pop(a)
                except ValueError:
                    break

        # get location of 2D fiducials using the indices
        image_idx = slicer.util.getNode('Image_idx')
        image_idx = slicer.util.arrayFromVolume(image_idx)
        # cv2.imwrite('C:/d/CTRegistrationHandEye/CTRegistrationHandEye/gradient.png', image_idx[0])


        #for some reason index 60 is empty, idk why, just did this to make it run
        # indices.pop(60)
        # pZ_3D.pop(60)
        # pX_3D.pop(60)
        # pY_3D.pop(60)

        pX_2D = []
        pY_2D = []

        # gradient = cv2.imread('C:/d/CTRegistrationHandEye/CTRegistrationHandEye/gradient.png')
        #
        # points_2D = np.where(image_idx == indices[1])
        #
        # print(indices[1])
        # for j in range(np.shape(points_2D)[1]):
        #     cv2.circle(gradient, (points_2D[1][j], points_2D[2][j]), 2, (0, 0, 255), 3)
        #
        # cv2.imwrite('C:/d/CTRegistrationHandEye/CTRegistrationHandEye/gradient2.png', gradient)

        for i in range(len(indices)):
            points_2D = np.where(image_idx == indices[i])

            # for j in range(np.shape(points_2D)[1]):
            #
            #     # points_2D[1][j], points_2D[2][j]
            #     cv2.circle(image_idx[0], (points_2D[1][j], points_2D[2][j]), 2, (0, 0, 255), 3)

            x = sum(points_2D[1]) / len(points_2D[1])
            pX_2D.append(x)


            y = sum(points_2D[2]) / len(points_2D[2])
            pY_2D.append(y)


        # cv2.imwrite('C:/d/CTRegistrationHandEye/CTRegistrationHandEye/brain_image_points2.png', image_idx[0])

        # format 3D and 3D coords
        P_3D = np.vstack((pX_3D, pY_3D, pZ_3D))
        P_3D = P_3D * np.vstack((-1, -1, 1))
        P_2D = np.vstack((pX_2D, pY_2D))


        print('3D points = \n', P_3D, '\n')
        print('2D points = \n', P_2D.shape, '\n')

        M_int_est, M_ext_est, M_proj, fre = self.cameraCombinedCalibration2(P_2D, P_3D)
        print('intrinsic matrix = \n', M_int_est, '\n')
        print('extrinsic matrix = \n', M_ext_est, '\n')
        print('projection matrix = \n', M_proj, '\n')
        print('fre = \n', fre, '\n')

        brain = cv2.imread('C:/d/CTRegistrationHandEye/CTRegistrationHandEye/brain_image.png')

        pixels, pixelErrors = self.PixelValidation(M_ext_est, P_3D, P_2D, M_int_est)
        # print("3D points Projected Into 2D = \n", len(pixels), "\n")
        print( "2D Reprojection Errors (pixels) = \n", pixelErrors, "\n")
        # print("Average 2D Reprojection Errors (pixels) = \n", np.average(pixelErrors), "\n")

        pixelsX = []
        pixelsY = []
        for i in range(np.shape(pixels)[0]):
            # print((pixels[i][0][0], pixels[i][1][0]))
            pixelsX.append(pixels[i][0][0])
            pixelsY.append(pixels[i][1][0])

            cv2.circle(brain, (round(pixels[i][0][0]), round(pixels[i][1][0])), 1, (0, 225,0), 3)

        # cv2.imwrite('C:/d/CTRegistrationHandEye/CTRegistrationHandEye/brain.png', brain)

        print("3D points Projected Into 2D = \n", pixelsX, "\n", pixelsY, "\n")
        print("3D points Projected Into 2D = \n", pixels, "\n")


        self.main(M_int_est, M_ext_est, P_2D, P_3D)


#######################################################################

#
# CTRegistrationHandEyeTest
#

class CTRegistrationHandEyeTest(ScriptedLoadableModuleTest):
    """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
    """
        self.setUp()
        self.test_CTRegistrationHandEye1()

    def test_CTRegistrationHandEye1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('CTRegistrationHandEye1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = CTRegistrationHandEyeLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
