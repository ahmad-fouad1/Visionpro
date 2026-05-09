from visionpro.recognition.matcher import matcher
from visionpro.recognition.descriptor import descriptor
def transformed_object(img1, img2, threshold=15):
    # Step 1: descriptors
    kp1, desc1 = descriptor(img1)
    kp2, desc2 = descriptor(img2)

    # safety check
    if desc1 is None or desc2 is None:
        return "Object Not Found"

    # Step 2: match
    matches = matcher(desc1, desc2)

    # Step 3: decision
    if len(matches) > threshold:
        return "Object Found"
    else:
        return "Object Not Found"