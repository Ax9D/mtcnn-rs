# mtcnn-rs
Rust implementation of the MTCNN face detector.

# Usage

```rust
    let mtcnn = MTCNN::new("mtcnn.pb").unwrap();//Load the model
    let image = imread("faces.jpg", IMREAD_COLOR).unwrap();
    let faces_roi=mtcnn.detect_faces_aligned(&image).unwrap(); //Returns a vec of cropped faces 

    imshow("Detected Face", &faces_roi[0]).unwrap();

    wait_key(0).unwrap();
```
   
# References
Original python implementation: https://github.com/ipazc/mtcnn
