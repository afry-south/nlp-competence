import numpy as np
import cv2
import pathlib

def get_opencv_img_from_buffer(buffer, flags=0):
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv2.imdecode(bytes_as_np_array, flags)

# Note Path() does not work, but needs to stringify!
def load_net():
    path_east = pathlib.Path(__file__).parent.absolute() / "frozen_east_text_detection.pb"
    net = cv2.dnn.readNet(str(path_east)) # raise TODO("Make use of cv2 neural net loader (`cv2.dnn.readNet`) and given path previously")
    return net

def decode_predictions(scores, geometry, min_confidence):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			uncertain_prediction = scoresData[x] < min_confidence
			if uncertain_prediction: continue

			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			(cos, sin) = (np.cos(angle), np.sin(angle))

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	return (rects, confidences)

def predict(img, net=load_net(), config=("-l eng --oem 1 --psm 7"), layers=["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"], min_confidence=0.5, padding=0.0):
    (newW, newH) = (320, 320)
    orig_cv2 = pil_to_cv2(img)
    
    (rW, rH) = (img.width / float(newW), img.height / float(newH))
    cv2_img = pil_to_cv2(img.resize((newW, newH)))
    resized_cv2_img = cv2_img_to_blob(cv2_img)
    net.setInput(resized_cv2_img)

    (scores, geometry) = net.forward(layers)
    (rects, confidences) = decode_predictions(scores, geometry, min_confidence)
    bboxes = non_max_suppression(np.array(rects), probs=confidences) # [(startX, startY, endX, endY), ...]
    
    results = []
    for (startX, startY, endX, endY) in bboxes:
        # Scale BBox
        (startX, startY, endX, endY) = (int(startX * rW), int(startY * rH), int(endX * rW), int(endY * rH))

        # Currently padding=0
        (dX, dY) = (int((endX - startX) * padding), int((endY - startY) * padding))

        # apply padding to each side of the bounding box, respectively
        (startX, startY) = (max(0, startX - dX), max(0, startY - dY))
        (endX, endY) = (min(img.width, endX + (dX * 2)), min(img.height, endY + (dY * 2)))


        # extract the actual padded ROI
        roi = orig_cv2[startY:endY, startX:endX]

        # Tesseract for OCR
        text = pytesseract.image_to_string(Image.fromarray(roi), config=config)
        # add the bounding box coordinates and OCR'd text to the list
        results.append(((startX, startY, endX, endY), text))

    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r:r[0][1])
    
    # loop over the results
    for ((startX, startY, endX, endY), text) in results:
        # display the text OCR'd by Tesseract
        print("OCR TEXT")
        print("========")
        print("{}\n".format(text))

        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        output = orig.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(output, text, (startX, startY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # show the output image
        cv2.imshow("Text Detection", output)
        cv2.waitKey(0)

    return results

net = load_net()
