#include <string>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <numeric> // for std::accumulate
#include <algorithm> // for std::shuffle
#include <random>
#include <iomanip>

constexpr int kNumClasses = 10;//类别数
constexpr int kImageSize = 784;//展开后列数

// 函数用于获取数据集目录
const std::string get_data_dir()
{
#ifdef _MSC_VER
    // 如果是Windows系统，返回数据集所在的目录路径
    return "D:/study/计算思维/大作业/代码";
#endif
}

// 数据集类，用于加载和存储图像数据和标签
class DataSet
{
public:
    // 构造函数，接收图像文件名和标签文件名作为参数
    DataSet(const std::string& image_filename, const std::string& label_filename)
    {
        load_images(image_filename); // 加载图像数据
        load_labels(label_filename); // 加载标签数据
        flatten_images(); // 展平图像数据，用于SVM
    }
    void flatten_images();//展平函数
    // 存储图像数据的向量，每个图像是一个cv::Mat对象
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> normalize_images;//储存归一化图像数据
    std::vector<std::vector<float>> feature_vectors; // 储存展平后的特征向量
    // 存储标签数据的向量
    std::vector<uint8_t> labels;
    std::vector<std::vector<float>> OneHotlabels;//储存onehot标签

private:
    // 用于存储读取的原始图像数据的缓冲区
    std::vector<char> buffer;
    // 加载图像数据的私有成员函数
    void load_images(const std::string& filename);
    // 加载标签数据的私有成员函数
    void load_labels(const std::string& filename);
    
};

//-----------------------------------------------------------------------------
//展平函数
void DataSet::flatten_images()
{
    feature_vectors.resize(normalize_images.size());
    for (size_t i = 0; i < normalize_images.size(); ++i)
    {
        cv::Mat flat_image = normalize_images[i].reshape(1, 1);
        std::vector<float> vec;
        vec.assign((float*)flat_image.datastart, (float*)flat_image.dataend);
        feature_vectors[i] = vec;
    }
}

// 从DataSet中提取前N个样本，返回新DataSet（只用于参数搜索）
DataSet get_subset(const DataSet& full, int n) {
    DataSet subset = full;
    if (n < full.images.size()) {
        subset.images.resize(n);
        subset.normalize_images.resize(n);
        subset.labels.resize(n);
        subset.OneHotlabels.resize(n);
        subset.flatten_images(); // 放在最后
    }
    return subset;
}


//数据集以二进制格式存储，需要确定字节序以正确解析文件
// 枚举类型，表示字节序（大端或小端）
enum class Endian
{
    LSB = 0, // 小端字节序
    MSB = 1  // 大端字节序
};

// 函数用于确定系统的字节序
static Endian get_endian()
{
    unsigned int num = 1;
    char* byte = (char*)&num;

    // 如果最低有效字节是1，则为小端字节序
    if (*byte)
        return Endian::LSB;
    else
        return Endian::MSB;
}

// 模板函数，用于交换字节序
template<typename T>
inline T swap_endian(T src);

// 特化模板函数，用于交换int类型的字节序
template<>
inline int swap_endian<int>(int src)
{
    int p1 = (src & 0xFF000000) >> 24;
    int p2 = (src & 0x00FF0000) >> 8;
    int p3 = (src & 0x0000FF00) << 8;
    int p4 = (src & 0x000000FF) << 24;
    return p1 + p2 + p3 + p4;
}

// 函数用于从二进制文件中读取一个整数
static int read_int_from_binary(std::ifstream& in)
{
    static Endian endian = get_endian(); // 获取系统的字节序

    int num;
    in.read(reinterpret_cast<char*>(&num), sizeof(num)); // 从文件中读取4个字节
    if (endian == Endian::LSB) // 如果是小端字节序，则交换字节序
        num = swap_endian(num);
    return num;
}

// 加载图像数据的成员函数
void DataSet::load_images(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary); // 以二进制模式打开文件
    if (!in.is_open()) {
        throw std::runtime_error("failed to open " + filename);
    }

    //读magic number
    char magic_bytes[4];
    in.read(magic_bytes, 4); // 读取魔数（文件类型标识）
    CV_Assert(magic_bytes[2] == 0x08); // 确认数据类型为unsigned byte
    CV_Assert(magic_bytes[3] == 3); // 确认数据维度为3D tensor

    //读dimensions
    int num_images = read_int_from_binary(in); // 读取图像数量
    int rows = read_int_from_binary(in); // 读取图像行数
    int cols = read_int_from_binary(in); // 读取图像列数

    //读data
    const size_t buffer_size = num_images * rows * cols; // 计算缓冲区大小
    buffer.resize(buffer_size);
    in.read(buffer.data(), buffer_size);


    // 初始化图像向量
    images = std::vector<cv::Mat>(num_images, cv::Mat(rows, cols, CV_8UC1));
    normalize_images = std::vector<cv::Mat>(num_images, cv::Mat(rows, cols, CV_32FC1)); // 归一化图像向量

    for (int i = 0; i < num_images; i++)
    {
        // 先用buffer创建临时Mat，再clone，保证数据独立
        cv::Mat tmp(rows, cols, CV_8UC1, buffer.data() + i * rows * cols);
        images[i] = tmp.clone();
        normalize_images[i] = images[i].clone();
        normalize_images[i].convertTo(normalize_images[i], CV_32FC1, 1.0 / 255.0);
    }

}
  
// 转换标签为one-hot编码，CNN 要用
std::vector<std::vector<float>> convertLabelsOneHot(const std::vector<uint8_t>& labels, int numClasses) {
    std::vector<std::vector<float>> oneHotLabels(labels.size(), std::vector<float>(numClasses, 0.0));
    for (size_t i = 0; i < labels.size(); ++i) {
        oneHotLabels[i][labels[i]] = 1.0;
    }
    return oneHotLabels;
}

// 加载标签数据的成员函数
void DataSet::load_labels(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary); // 以二进制模式打开文件
    if (!in.is_open()) {
        throw std::runtime_error("failed to open " + filename);
    }


    char magic_bytes[4];
    in.read(magic_bytes, 4); // 读取魔数（文件类型标识）
    CV_Assert(magic_bytes[2] == 0x08); // 确认数据类型为unsigned byte
    CV_Assert(magic_bytes[3] == 1); // 确认数据维度为1D vector

    int num_labels = read_int_from_binary(in); // 读取标签数量
    labels = std::vector<uint8_t>(num_labels); // 初始化标签向量
    in.read(reinterpret_cast<char*>(labels.data()), num_labels); // 读取标签数据到向量中

    OneHotlabels = convertLabelsOneHot(labels, 10); // 实现标签one-hot编码
}


//训练SVM的函数
void train_svm(const DataSet& train_set, double C = 10, double gamma = 0.01)
{
    // 转换特征矩阵：vector<vector<float>> -> cv::Mat (N x 784)
    cv::Mat train_features(train_set.feature_vectors.size(), kImageSize, CV_32FC1);
    for (size_t i = 0; i < train_set.feature_vectors.size(); ++i) {
        cv::Mat(1, kImageSize, CV_32FC1, (float*)train_set.feature_vectors[i].data()).copyTo(train_features.row(i));
    }

    // 标签矩阵转换（显式类型转换）
    cv::Mat train_labels(train_set.labels.size(), 1, CV_32SC1);
    for (size_t i = 0; i < train_set.labels.size(); ++i) {
        train_labels.at<int>(i) = static_cast<int>(train_set.labels[i]);
    }

    // 添加类型验证
    CV_Assert(train_features.type() == CV_32FC1);
    CV_Assert(train_labels.type() == CV_32SC1);

    // 创建SVM模型
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);       // 多分类类型
    svm->setKernel(cv::ml::SVM::RBF);       // 高斯核（MNIST非线性可分）
    svm->setC(C);                          // 正则化参数（需交叉验证优化）
    svm->setGamma(gamma);                    // RBF核参数（与特征方差相关）

    // 训练模型（约需10-30分钟，依赖硬件）
    svm->train(train_features, cv::ml::ROW_SAMPLE, train_labels);

    // 保存模型
    svm->save("mnist_svm.xml");
}

// 计算并保存多分类指标和混淆矩阵
void save_metrics_and_scores(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    const std::vector<std::vector<float>>& scores, // 每个样本的每类得分
    const std::string& metrics_file,
    const std::string& scores_file)
{
    int num_classes = kNumClasses;
    std::vector<std::vector<int>> confusion(num_classes, std::vector<int>(num_classes, 0));
    for (size_t i = 0; i < y_true.size(); ++i)
        confusion[y_true[i]][y_pred[i]]++;

    // 计算每类的precision/recall/f1
    std::vector<double> precision(num_classes), recall(num_classes), f1(num_classes);
    int total_correct = 0;
    for (int c = 0; c < num_classes; ++c) {
        int tp = confusion[c][c];
        int fp = 0, fn = 0;
        for (int k = 0; k < num_classes; ++k) {
            if (k != c) {
                fp += confusion[k][c];
                fn += confusion[c][k];
            }
        }
        int denom_p = tp + fp, denom_r = tp + fn;
        precision[c] = denom_p ? (double)tp / denom_p : 0;
        recall[c] = denom_r ? (double)tp / denom_r : 0;
        f1[c] = (precision[c] + recall[c]) ? 2 * precision[c] * recall[c] / (precision[c] + recall[c]) : 0;
        total_correct += tp;
    }
    double accuracy = (double)total_correct / y_true.size();

    // 保存指标
    std::ofstream ofs(metrics_file);
    ofs << "accuracy," << accuracy << "\n";
    ofs << "class,precision,recall,f1\n";
    for (int c = 0; c < num_classes; ++c)
        ofs << c << "," << precision[c] << "," << recall[c] << "," << f1[c] << "\n";
    ofs << "confusion_matrix\n";
    for (int c = 0; c < num_classes; ++c) {
        for (int k = 0; k < num_classes; ++k)
            ofs << confusion[c][k] << (k == num_classes - 1 ? "\n" : ",");
    }
    ofs.close();

    // 保存每个样本的真实标签、预测标签、每类得分（用于python画ROC/AUC）
    std::ofstream ofs2(scores_file);
    ofs2 << "true_label,pred_label";
    for (int c = 0; c < num_classes; ++c) ofs2 << ",score_" << c;
    ofs2 << "\n";
    for (size_t i = 0; i < y_true.size(); ++i) {
        ofs2 << y_true[i] << "," << y_pred[i];
        for (int c = 0; c < num_classes; ++c)
            ofs2 << "," << scores[i][c];
        ofs2 << "\n";
    }
    ofs2.close();
}


//SVM模型测试
void test_svm(const DataSet& test_set)
{
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("mnist_svm.xml");
    cv::Mat test_features(test_set.feature_vectors.size(), kImageSize, CV_32FC1);
    for (size_t i = 0; i < test_set.feature_vectors.size(); ++i) {
        cv::Mat(1, kImageSize, CV_32FC1, (float*)test_set.feature_vectors[i].data()).copyTo(test_features.row(i));
    }

    // 预测标签
    cv::Mat predictions;
    svm->predict(test_features, predictions);

    // 获取每个样本对每个类别的决策分数（one-vs-rest）
    std::vector<std::vector<float>> all_scores(test_set.feature_vectors.size(), std::vector<float>(kNumClasses, 0.0f));
    for (int c = 0; c < kNumClasses; ++c) {
        cv::Ptr<cv::ml::SVM> svm_c = cv::ml::SVM::load("mnist_svm.xml");
        svm_c->setClassWeights(cv::Mat()); // 确保不带权重
        // 由于OpenCV多分类SVM不直接支持概率，这里用决策值近似
        cv::Mat dec_values;
        svm_c->predict(test_features, dec_values, cv::ml::StatModel::RAW_OUTPUT);
        for (int i = 0; i < dec_values.rows; ++i)
            all_scores[i][c] = dec_values.at<float>(i);
    }

    // 收集真实标签和预测标签
    std::vector<int> y_true, y_pred;
    for (int i = 0; i < predictions.rows; ++i) {
        y_true.push_back(static_cast<int>(test_set.labels[i]));
        y_pred.push_back(static_cast<int>(predictions.at<float>(i)));
    }

    // 保存指标和分数
    save_metrics_and_scores(y_true, y_pred, all_scores, "metrics.csv", "scores.csv");

    // 输出准确率
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
        if (y_true[i] == y_pred[i]) ++correct;
    std::cout << "SVM Accuracy: " << (correct * 100.0 / y_true.size()) << "%\n";
}

// 交叉验证函数，返回平均准确率
double cross_validate_svm(const DataSet& dataset, double C, double gamma, int k_folds = 5) {
    // 构造特征和标签矩阵
    int N = dataset.feature_vectors.size();
    cv::Mat features(N, kImageSize, CV_32FC1);
    for (int i = 0; i < N; ++i)
        cv::Mat(1, kImageSize, CV_32FC1, (float*)dataset.feature_vectors[i].data()).copyTo(features.row(i));
    cv::Mat labels(N, 1, CV_32SC1);
    for (int i = 0; i < N; ++i)
        labels.at<int>(i) = static_cast<int>(dataset.labels[i]);

    // 打乱索引
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42); // 固定种子保证可复现
    std::shuffle(indices.begin(), indices.end(), rng);

    int fold_size = N / k_folds;
    double total_acc = 0.0;

    for (int fold = 0; fold < k_folds; ++fold) {
        // 生成训练和验证索引
        std::vector<int> val_idx(indices.begin() + fold * fold_size,
            (fold == k_folds - 1) ? indices.end() : indices.begin() + (fold + 1) * fold_size);
        std::vector<int> train_idx;
        train_idx.reserve(N - val_idx.size());
        for (int i : indices) {
            if (std::find(val_idx.begin(), val_idx.end(), i) == val_idx.end())
                train_idx.push_back(i);
        }

        // 构造训练和验证集
        cv::Mat train_feat(train_idx.size(), kImageSize, CV_32FC1);
        cv::Mat train_lab(train_idx.size(), 1, CV_32SC1);
        for (size_t i = 0; i < train_idx.size(); ++i) {
            features.row(train_idx[i]).copyTo(train_feat.row(i));
            train_lab.at<int>(i) = labels.at<int>(train_idx[i]);
        }
        cv::Mat val_feat(val_idx.size(), kImageSize, CV_32FC1);
        cv::Mat val_lab(val_idx.size(), 1, CV_32SC1);
        for (size_t i = 0; i < val_idx.size(); ++i) {
            features.row(val_idx[i]).copyTo(val_feat.row(i));
            val_lab.at<int>(i) = labels.at<int>(val_idx[i]);
        }

        // 训练SVM
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setC(C);
        svm->setGamma(gamma);
        svm->train(train_feat, cv::ml::ROW_SAMPLE, train_lab);

        // 验证
        cv::Mat pred;
        svm->predict(val_feat, pred);
        int correct = 0;
        for (int i = 0; i < pred.rows; ++i) {
            if (static_cast<int>(pred.at<float>(i)) == val_lab.at<int>(i))
                ++correct;
        }
        total_acc += correct * 1.0 / val_idx.size();

        // 在for (int fold = 0; fold < k_folds; ++fold) {...} 内部加
        std::cout << "Fold " << fold << " acc: " << (correct * 1.0 / val_idx.size()) << std::endl;

    }
    return total_acc / k_folds;
}

int main()
{
    
    //1.数据集解析
    const std::string data_dir = get_data_dir(); // 获取数据集目录
    DataSet train_set(data_dir + "/train-images.idx3-ubyte", data_dir + "/train-labels.idx1-ubyte"); // 创建训练数据集对象
    DataSet test_set(data_dir + "/t10k-images.idx3-ubyte", data_dir + "/t10k-labels.idx1-ubyte"); // 创建测试数据集对象

    //2.数据集预处理
    //归一化和转化标签   在创建数据集时已实现归一化，one-hot编码标签
    

    //3.模型调用
    ////只用部分数据做参数搜索
    //int subset_size = 2000;
    //DataSet small_train_set = get_subset(train_set, subset_size);

    //std::vector<double> C_list = { 0.1, 1, 10, 100 };
    //std::vector<double> gamma_list = { 0.001, 0.01, 0.1, 1 };
    //double best_acc = 0;
    //double best_C = 1, best_gamma = 0.01;
    //for (double C : C_list) {
    //    for (double gamma : gamma_list) {
    //        double acc = cross_validate_svm(small_train_set, C, gamma, 3); // 3折更快
    //        std::cout << "C=" << C << ", gamma=" << gamma << ", CV acc=" << acc << std::endl;
    //        if (acc > best_acc) {
    //            best_acc = acc;
    //            best_C = C;
    //            best_gamma = gamma;
    //        }
    //    }
    //}
    //std::cout << "Best C=" << best_C << ", Best gamma=" << best_gamma << ", Best CV acc=" << best_acc << std::endl;

    // 用最优参数训练全量模型
    //train_svm(train_set, best_C, best_gamma);
    //std::cout << "SVM训练完成！" << std::endl;

    test_svm(test_set);
    return 0;
}
