
layoutManager = slicer.app.layoutManager()
view = layoutManager.threeDWidget(0).threeDView()
threeDViewNode = view.mrmlViewNode()
cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(threeDViewNode)
renderWindow = view.renderWindow()
renderers = renderWindow.GetRenderers()
renderer = renderers.GetItemAsObject(0)
camera = cameraNode.GetCamera()

reader = vtk.vtkPNGReader()
#reader.SetFileName("C:/d/CTRegistrationHandEye/CTRegistrationHandEye/brain.png")
reader.SetFileName("C:/d/CTRegistrationHandEye/CTRegistrationHandEye/brain_image_small_2Dreprojection.png")
reader.Update()

import numpy as np
intrinsic = np.array([[412.93036447, 0, 828.36842508],[0, 421.34767043, 789.78228686],[0, 0, 1]]) 
extrinsic = np.array( [[-3.15662473e-01, 9.28358025e-01, 1.96236032e-01, 6.39231514e+01],[8.43356646e-01, 2.00660954e-01, 4.98482447e-01, 1.17914990e+02],[4.23461036e-01, 3.22900836e-01, -8.46413493e-01, 3.62257338e+02],[0, 0, 0, 1]])

cx = intrinsic[0, 2]
cy = intrinsic[1, 2]
f = intrinsic[0, 0]
h = 1080
w = 1349

import mathwhere
wcx = -2 * (cx - float(w) / 2) / w
wcy = 2 * (cy - float(h) / 2) / h
camera.SetWindowCenter(wcx, wcy)

view_angle = 180 / math.pi * (2.0 * math.atan2(h / 2.0, f))
camera.SetViewAngle(view_angle)

camera.SetPosition(0, 0, 0)
camera.SetFocalPoint(0, 0, f)
camera.SetViewUp(0, -1, 0)

texture = vtk.vtkTexture()
texture.SetInputConnection(reader.GetOutputPort())
renderer.SetTexturedBackground(True)
renderer.SetBackgroundTexture(texture)

P_3D = np.array([[-251, -252, -236, -248, -234, -234, -241, -255, -238, -242, -232, -230, -229, -235, -222, -227, -223, -220, -240, -237, -245, -222, -216, -243, -222, -222, -212, -250, -237, -238, -210, -215, -214, -216, -221, -226, -218, -221, -217],[-208, -211, -214, -205, -219, -205, -209, -181, -189, -194, -200, -203, -207, -190, -223, -198, -215, -220, -181, -183, -174, -203, -213, -172, -199, -197, -210, -197, -168, -175, -203, -209, -208, -194, -195, -179, -182, -193, -189],[254, 254, 254, 255, 255, 256, 256, 257, 257, 257, 257, 257, 257, 258, 258, 259, 259, 259, 260, 260, 261, 261, 261, 263, 263, 264, 264, 265, 266, 266, 266, 267, 268, 270, 271, 274, 278, 278, 279]])


P_3D_ExtX = []
P_3D_ExtY = []
P_3D_ExtZ = []


for k in range(P_3D.shape[1]):
   point = P_3D[:, k]
   point = np.reshape(point, (3, 1))

   point = np.vstack((point, 1))

   # Register 3D point to line
   cameraPoint = (extrinsic * -1) @ point

   P_3D_ExtX.append(cameraPoint[0][0])
   P_3D_ExtY.append(cameraPoint[1][0])
   P_3D_ExtZ.append(cameraPoint[2][0])


markupsNode_3D = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
markupsNode_3D.SetName("3DFiducials")

for i in range(np.shape(P_3D)[1]):
   x = np.hstack((P_3D_ExtX[i], P_3D_ExtY[i], P_3D_ExtZ[i]))
   markupsNode_3D.AddFiducialFromArray(x)

