use std::{ffi::c_void, fs::File, io::Read, path::Path};

use opencv::{core::*};
use tensorflow as tf;
use tf::*;
use std::error::Error;
use std::result::Result;

pub fn read_binary_proto<T: AsRef<str>>(path_str: T) -> Result<Vec<u8>, Box<dyn Error>> {
    let path = Path::new(path_str.as_ref());
    
    let mut file = File::open(&path)?;
    let mut s = Vec::new();

    file.read_to_end(&mut s)?;
    
    Ok(s)
}


pub struct MTCNN {
    session: tf::Session,
    min_size_data: tf::Tensor<f32>,
    thresholds_data: tf::Tensor<f32>,
    factor_data: tf::Tensor<f32>,

    thresholds_op: tf::Operation,
    min_size_op: tf::Operation,
    factor_op: tf::Operation,

    input_op: tf::Operation,
    box_op: tf::Operation,
    prob_op: tf::Operation,
}

impl MTCNN {
    pub fn new<T: AsRef<str>>(modelPath: T) -> Result<Self, Box<dyn Error>> {
        let graph_def = read_binary_proto(modelPath)?;

        let mut graph = tf::Graph::new();
        graph
            .import_graph_def(graph_def.as_slice(), &tf::ImportGraphDefOptions::new())?;

        let session = Session::new(&SessionOptions::new(), &graph)?;

        let min_size_data = Tensor::new(&[]).with_values(&[40f32])?;
        let thresholds_data = Tensor::new(&[3])
            .with_values(&[0.6f32, 0.7f32, 0.7f32])
            .unwrap();

        let factor_data = Tensor::new(&[]).with_values(&[0.709f32]).unwrap();

        let min_size_op = graph.operation_by_name_required("min_size")?;
        let thresholds_op = graph.operation_by_name_required("thresholds")?;
        let factor_op = graph.operation_by_name_required("factor")?;

        let input_op = graph.operation_by_name_required("input")?;

        let box_op = graph.operation_by_name_required("box")?;
        let prob_op = graph.operation_by_name_required("prob")?;

        Ok(Self {
            session,
            min_size_data,
            thresholds_data,
            factor_data,

            min_size_op,
            thresholds_op,
            factor_op,

            input_op,
            box_op,
            prob_op,
        })
    }
    fn create_image_tensor(image: &Mat) -> tf::Tensor<f32> {
        //image in BGR u8 format

        let im_row = image.rows();
        let im_cols = image.cols();
        let mut image_tensor = tf::Tensor::<f32>::new(&[im_row as u64, im_cols as u64, 3]);

        let mut image_cv_mat = unsafe {
            let tensor_ptr = image_tensor.as_mut_ptr();

            match Mat::new_rows_cols_with_data(im_row, im_cols, CV_32FC3, tensor_ptr as *mut c_void, 0) {
                Ok(it) => it,
                _ => unreachable!(),
            }
        };
        //copy_to(image,&mut imageCVMat,&no_array().unwrap()).unwrap();
        image
            .convert_to(&mut image_cv_mat, CV_32FC3, 1.0, 0.0)
            .unwrap();

        image_tensor
    }
    pub fn evaluate(&self, image: tf::Tensor<f32>) -> Result<(tf::Tensor<f32>, tf::Tensor<f32>), Box<dyn Error>>{
        let mut args = SessionRunArgs::new();

        //Load our parameters for the model
        args.add_feed(&self.min_size_op, 0, &self.min_size_data);
        args.add_feed(&self.thresholds_op, 0, &self.thresholds_data);
        args.add_feed(&self.factor_op, 0, &self.factor_data);

        //Load our input image
        args.add_feed(&self.input_op, 0, &image);

        let box_fetch_token = args.request_fetch(&self.box_op, 0);
        let prob_fetch_token = args.request_fetch(&self.prob_op, 0);

        self.session.run(&mut args)?;

        let bbox_tensor: tf::Tensor<f32> = args.fetch(box_fetch_token)?;
        let prob_tensor: tf::Tensor<f32> = args.fetch(prob_fetch_token)?;

        Ok(
        (bbox_tensor, prob_tensor)
        )
    }
    fn align_rect(rect: &mut Rect, width: i32, height: i32) {
        rect.x = rect.x.max(0);
        rect.y = rect.y.max(0);

        if rect.x + rect.width > width {
            rect.width = width - rect.x;
        }

        if rect.y + rect.height > height {
            rect.height = height - rect.y;
        }
    }
    pub fn detect_faces_aligned(&self, image: &Mat) -> Result<Vec<Mat>, Box<dyn Error>> {
        let image_tensor = Self::create_image_tensor(&image);

        let (bboxes, _probs) = self.evaluate(image_tensor)?;

        //println!("{}",bboxes);

        let mut faces = Vec::<Mat>::new();

        for bbox in bboxes.chunks_exact(4) {
            let (y1, x1, y2, x2) = (
                bbox[0] as i32,
                bbox[1] as i32,
                bbox[2] as i32,
                bbox[3] as i32,
            );

            let mut face_rect = Rect::new(x1, y1, x2 - x1, y2 - y1);

            Self::align_rect(&mut face_rect, image.cols(), image.rows());

            faces.push(Mat::roi(&image, face_rect)?);
        }

        Ok(
            faces
        )
    }
}

#[cfg(test)]
mod tests {
    use opencv::{highgui::{imshow, wait_key}, imgcodecs::{IMREAD_COLOR, imread}};

    use crate::MTCNN;

    #[test]
    fn basic_usage() {
        let mtcnn = MTCNN::new("mtcnn.pb").unwrap();
        let image = imread("faces.jpg", IMREAD_COLOR).unwrap();
        let faces_roi=mtcnn.detect_faces_aligned(&image).unwrap();   

        imshow("Detected Face", &faces_roi[0]).unwrap();

        wait_key(0).unwrap();
    }
}