
def process_images(img1, img2, detector, app):

  """
  Calculate the similarity score between the two images.

  Args:
    image1_path : str, path of first image
    image1_path : str, path of second image

  Returns:
    float, similarity score

  """

  # to prepare images

  faces_1 = app.get(img1)[0]['embedding']
  faces_2 = app.get(img2)[0]['embedding']

  score =  detector.compute_sim(faces_1, faces_2)
  # to compare image similarity
  print(f'Similarity score: {score}')
  return score
