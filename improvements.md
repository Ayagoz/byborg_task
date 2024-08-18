# Benchmark descussion: 

I updated the `create_dummy_tags.py` to generate random number of tags. 
Following question shows that the running time depends on different solutions we use in our service infrastructure. 
If we have millions of data + a lot of the requests it must be a different approaches.
In code presented 3 main types of inference runs: batch, online and offline. 

## 1. Infrastructure:
- Do we store all tags embedding in memory?
- Do we do online or batch inference? 
- Do we load samples for inference from memory? Or do we receive other ways?
- What type of GPU we have? How many embeddings is suitable to store? 
- Do we calculate embeddings and similarity metrics on the same device?

## 2. Data:
- Do we have a hierarchical structure of tags?
- What should be the cost of the calculation and how precise the prediction should be?

## 3. Optimization:
- We can use numpy's upgrade Numba and operate with numpy only to reduce time on transfering vectors from device to device. It allows to calculate similarity mertrics using numpy too. 
- We can use torch and calculate metrics on torch. 


# Main task: Size Optimization (See param embeddings_dim in file .env)
## 1. How to Optimize Size of Embedding Vectors:
- **Classic Dimensionality Reduction Techniques:** PCA, t-SNE, SVD.
- **Deep Learning Size OptimizationP:**  Autoencoders, Quantization, Half-Precision. 

## 2. How to Measure Performance of Reduced Vectors:
- **Accuracy Comparison:** Measure the accuracy of search results before and after reduction. Compare the overlap in top N results.
- **Execution Time:** Benchmark the time taken to search using the original vs. reduced embeddings.
- **Memory Usage:** Track the memory usage before and after the reduction.

## 3. CICD for system: 
- **Unit test:** To test API.
- **Integration test:** Validate that the reduced embeddings still provide accurate search results. Ensure that the reduction technique meets the performance criteria. Set metrics and known test data + test cases to validate automatically.


# Main task: Improvements and Optimizations

## 1. Code Optimizations
- **Vectorization:** Utilize vectorized operations with NumPy or similar libraries to improve the speed of computations.
- **Batch Processing:** For large datasets, implement batch processing to handle computations more efficiently.
- **Caching:** Implement caching mechanisms to store frequently accessed results, reducing redundant calculations.

## 2. Maintenance Improvements
- **Modularization:** Break down the code into smaller, reusable modules to improve maintainability and readability.
- **Testing:** Implement unit tests and integration tests to ensure the reliability of the system.
- **Documentation:** Provide clear and concise documentation for each module and function to assist future developers.

## 3. Deployment Strategies
- **CI/CD Pipeline:** Implement a CI/CD pipeline to automate testing, building, and deployment processes.
- **Scalability:** Consider using Kubernetes for deploying the application in a scalable manner across multiple nodes.

## 4. Additional Improvements
- **Logging and Monitoring:** Implement logging and monitoring to track system performance and troubleshoot issues.

