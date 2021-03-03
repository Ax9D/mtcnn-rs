# mtcnn-rs
Rust implementation of the MTCNN face detector.

# Usage

```rust
    let  mtcnn  =  MTCNN::new("mtcnn.pb").unwrap();//Load the model
    let  image  =  imread("faces.jpg", IMREAD_COLOR).unwrap();
    let  facesROI  = mtcnn.detectFacesAligned(&image).unwrap();//Returns a vec of cropped faces
    imshow("Detected Face", &facesROI[0]).unwrap();
    wait_key(0).unwrap();
```
   
# References
Original python implementation: https://github.com/ipazc/mtcnn
