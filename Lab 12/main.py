import cv2
import class_coin_detection

def main():
    counter = class_coin_detection.Coin_detection()
    video = cv2.VideoCapture(0)  # you maybe need to adjust the index here

    while True:
        ret, frame = video.read()

        change, roi = counter.run_main(frame)
        if change is None:
            change = 0.0
        cv2.putText(
            roi,
            "Total: " + str(round(change, 2)) + " CHF",
            (14, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )


        cv2.imshow('Detected Swiss Coins', roi) # Replace none by image

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


