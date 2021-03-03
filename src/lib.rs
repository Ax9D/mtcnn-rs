use std::ffi::c_void;

use crate::utils;
use opencv::{core::*, imgproc::rectangle};
use tensorflow as tf;
use tf::*;
pub struct MTCNN {
    session: tf::Session,
    min_sizeData: tf::Tensor<f32>,
    thresholdsData: tf::Tensor<f32>,
    factorData: tf::Tensor<f32>,

    thresholdsOp: tf::Operation,
    min_sizeOp: tf::Operation,
    factorOp: tf::Operation,

    inputOp: tf::Operation,
    boxOp: tf::Operation,
    probOp: tf::Operation,
}

impl MTCNN {
    pub fn new(modelPath: &String) -> Self {
        let graph_def = utils::tf::readBinaryProto(modelPath);

        let mut graph = tf::Graph::new();
        graph
            .import_graph_def(graph_def.as_slice(), &tf::ImportGraphDefOptions::new())
            .unwrap();

        let session = Session::new(&SessionOptions::new(), &graph).unwrap();

        let min_sizeData = Tensor::new(&[]).with_values(&[40f32]).unwrap();
        let thresholdsData = Tensor::new(&[3])
            .with_values(&[0.6f32, 0.7f32, 0.7f32])
            .unwrap();
        let factorData = Tensor::new(&[]).with_values(&[0.709f32]).unwrap();

        let min_sizeOp = graph.operation_by_name_required("min_size").unwrap();
        let thresholdsOp = graph.operation_by_name_required("thresholds").unwrap();
        let factorOp = graph.operation_by_name_required("factor").unwrap();

        let inputOp = graph.operation_by_name_required("input").unwrap();

        let boxOp = graph.operation_by_name_required("box").unwrap();
        let probOp = graph.operation_by_name_required("prob").unwrap();

        Self {
            session,
            min_sizeData,
            thresholdsData,
            factorData,

            min_sizeOp,
            thresholdsOp,
            factorOp,

            inputOp,
            boxOp,
            probOp,
        }
    }
    fn createImageTensor(image: &Mat) -> tf::Tensor<f32> {
        //image in BGR u8 format

        let imRow = image.rows();
        let imCols = image.cols();
        let mut imageTensor = tf::Tensor::<f32>::new(&[imRow as u64, imCols as u64, 3]);

        let mut imageCVMat = unsafe {
            let tensorPtr = imageTensor.as_mut_ptr();

            Mat::new_rows_cols_with_data(imRow, imCols, CV_32FC3, tensorPtr as *mut c_void, 0)
                .unwrap()
        };
        //copy_to(image,&mut imageCVMat,&no_array().unwrap()).unwrap();
        image
            .convert_to(&mut imageCVMat, CV_32FC3, 1.0, 0.0)
            .unwrap();

        imageTensor
    }
    pub fn evaluate(&self, image: tf::Tensor<f32>) -> (tf::Tensor<f32>, tf::Tensor<f32>) {
        let mut args = SessionRunArgs::new();

        //Load our parameters for the model
        args.add_feed(&self.min_sizeOp, 0, &self.min_sizeData);
        args.add_feed(&self.thresholdsOp, 0, &self.thresholdsData);
        args.add_feed(&self.factorOp, 0, &self.factorData);

        //Load our input image
        args.add_feed(&self.inputOp, 0, &image);

        let boxFetchToken = args.request_fetch(&self.boxOp, 0);
        let probFetchToken = args.request_fetch(&self.probOp, 0);

        self.session.run(&mut args).unwrap();

        let bboxTensor: tf::Tensor<f32> = args.fetch(boxFetchToken).unwrap();
        let probTensor: tf::Tensor<f32> = args.fetch(probFetchToken).unwrap();

        (bboxTensor, probTensor)
    }
    fn alignRect(rect: &mut Rect, width: i32, height: i32) {
        rect.x = rect.x.max(0);
        rect.y = rect.y.max(0);

        if rect.x + rect.width > width {
            rect.width = width - rect.x;
        }

        if rect.y + rect.height > height {
            rect.height = height - rect.y;
        }
    }
    pub fn detectFacesAligned(&self, image: &Mat) -> Vec<Mat> {
        let imageTensor = Self::createImageTensor(&image);

        let (bboxes, probs) = self.evaluate(imageTensor);

        //println!("{}",bboxes);

        let mut faces = Vec::<Mat>::new();

        for bbox in bboxes.chunks_exact(4) {
            let (y1, x1, y2, x2) = (
                bbox[0] as i32,
                bbox[1] as i32,
                bbox[2] as i32,
                bbox[3] as i32,
            );

            let mut faceRect = Rect::new(x1, y1, x2 - x1, y2 - y1);

            Self::alignRect(&mut faceRect, image.cols(), image.rows());

            faces.push(Mat::roi(&image, faceRect).unwrap());
        }
        faces
    }
}
