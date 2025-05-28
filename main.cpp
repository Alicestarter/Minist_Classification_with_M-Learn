#include <string>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <numeric> // for std::accumulate
#include <algorithm> // for std::shuffle
#include <random>
#include <iomanip>

constexpr int kNumClasses = 10;//�����
constexpr int kImageSize = 784;//չ��������

// �������ڻ�ȡ���ݼ�Ŀ¼
const std::string get_data_dir()
{
#ifdef _MSC_VER
    // �����Windowsϵͳ���������ݼ����ڵ�Ŀ¼·��
    return "D:/study/����˼ά/����ҵ/����";
#endif
}

// ���ݼ��࣬���ڼ��غʹ洢ͼ�����ݺͱ�ǩ
class DataSet
{
public:
    // ���캯��������ͼ���ļ����ͱ�ǩ�ļ�����Ϊ����
    DataSet(const std::string& image_filename, const std::string& label_filename)
    {
        load_images(image_filename); // ����ͼ������
        load_labels(label_filename); // ���ر�ǩ����
        flatten_images(); // չƽͼ�����ݣ�����SVM
    }
    void flatten_images();//չƽ����
    // �洢ͼ�����ݵ�������ÿ��ͼ����һ��cv::Mat����
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> normalize_images;//�����һ��ͼ������
    std::vector<std::vector<float>> feature_vectors; // ����չƽ�����������
    // �洢��ǩ���ݵ�����
    std::vector<uint8_t> labels;
    std::vector<std::vector<float>> OneHotlabels;//����onehot��ǩ

private:
    // ���ڴ洢��ȡ��ԭʼͼ�����ݵĻ�����
    std::vector<char> buffer;
    // ����ͼ�����ݵ�˽�г�Ա����
    void load_images(const std::string& filename);
    // ���ر�ǩ���ݵ�˽�г�Ա����
    void load_labels(const std::string& filename);
    
};

//-----------------------------------------------------------------------------
//չƽ����
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

// ��DataSet����ȡǰN��������������DataSet��ֻ���ڲ���������
DataSet get_subset(const DataSet& full, int n) {
    DataSet subset = full;
    if (n < full.images.size()) {
        subset.images.resize(n);
        subset.normalize_images.resize(n);
        subset.labels.resize(n);
        subset.OneHotlabels.resize(n);
        subset.flatten_images(); // �������
    }
    return subset;
}


//���ݼ��Զ����Ƹ�ʽ�洢����Ҫȷ���ֽ�������ȷ�����ļ�
// ö�����ͣ���ʾ�ֽ��򣨴�˻�С�ˣ�
enum class Endian
{
    LSB = 0, // С���ֽ���
    MSB = 1  // ����ֽ���
};

// ��������ȷ��ϵͳ���ֽ���
static Endian get_endian()
{
    unsigned int num = 1;
    char* byte = (char*)&num;

    // ��������Ч�ֽ���1����ΪС���ֽ���
    if (*byte)
        return Endian::LSB;
    else
        return Endian::MSB;
}

// ģ�庯�������ڽ����ֽ���
template<typename T>
inline T swap_endian(T src);

// �ػ�ģ�庯�������ڽ���int���͵��ֽ���
template<>
inline int swap_endian<int>(int src)
{
    int p1 = (src & 0xFF000000) >> 24;
    int p2 = (src & 0x00FF0000) >> 8;
    int p3 = (src & 0x0000FF00) << 8;
    int p4 = (src & 0x000000FF) << 24;
    return p1 + p2 + p3 + p4;
}

// �������ڴӶ������ļ��ж�ȡһ������
static int read_int_from_binary(std::ifstream& in)
{
    static Endian endian = get_endian(); // ��ȡϵͳ���ֽ���

    int num;
    in.read(reinterpret_cast<char*>(&num), sizeof(num)); // ���ļ��ж�ȡ4���ֽ�
    if (endian == Endian::LSB) // �����С���ֽ����򽻻��ֽ���
        num = swap_endian(num);
    return num;
}

// ����ͼ�����ݵĳ�Ա����
void DataSet::load_images(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary); // �Զ�����ģʽ���ļ�
    if (!in.is_open()) {
        throw std::runtime_error("failed to open " + filename);
    }

    //��magic number
    char magic_bytes[4];
    in.read(magic_bytes, 4); // ��ȡħ�����ļ����ͱ�ʶ��
    CV_Assert(magic_bytes[2] == 0x08); // ȷ����������Ϊunsigned byte
    CV_Assert(magic_bytes[3] == 3); // ȷ������ά��Ϊ3D tensor

    //��dimensions
    int num_images = read_int_from_binary(in); // ��ȡͼ������
    int rows = read_int_from_binary(in); // ��ȡͼ������
    int cols = read_int_from_binary(in); // ��ȡͼ������

    //��data
    const size_t buffer_size = num_images * rows * cols; // ���㻺������С
    buffer.resize(buffer_size);
    in.read(buffer.data(), buffer_size);


    // ��ʼ��ͼ������
    images = std::vector<cv::Mat>(num_images, cv::Mat(rows, cols, CV_8UC1));
    normalize_images = std::vector<cv::Mat>(num_images, cv::Mat(rows, cols, CV_32FC1)); // ��һ��ͼ������

    for (int i = 0; i < num_images; i++)
    {
        // ����buffer������ʱMat����clone����֤���ݶ���
        cv::Mat tmp(rows, cols, CV_8UC1, buffer.data() + i * rows * cols);
        images[i] = tmp.clone();
        normalize_images[i] = images[i].clone();
        normalize_images[i].convertTo(normalize_images[i], CV_32FC1, 1.0 / 255.0);
    }

}
  
// ת����ǩΪone-hot���룬CNN Ҫ��
std::vector<std::vector<float>> convertLabelsOneHot(const std::vector<uint8_t>& labels, int numClasses) {
    std::vector<std::vector<float>> oneHotLabels(labels.size(), std::vector<float>(numClasses, 0.0));
    for (size_t i = 0; i < labels.size(); ++i) {
        oneHotLabels[i][labels[i]] = 1.0;
    }
    return oneHotLabels;
}

// ���ر�ǩ���ݵĳ�Ա����
void DataSet::load_labels(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary); // �Զ�����ģʽ���ļ�
    if (!in.is_open()) {
        throw std::runtime_error("failed to open " + filename);
    }


    char magic_bytes[4];
    in.read(magic_bytes, 4); // ��ȡħ�����ļ����ͱ�ʶ��
    CV_Assert(magic_bytes[2] == 0x08); // ȷ����������Ϊunsigned byte
    CV_Assert(magic_bytes[3] == 1); // ȷ������ά��Ϊ1D vector

    int num_labels = read_int_from_binary(in); // ��ȡ��ǩ����
    labels = std::vector<uint8_t>(num_labels); // ��ʼ����ǩ����
    in.read(reinterpret_cast<char*>(labels.data()), num_labels); // ��ȡ��ǩ���ݵ�������

    OneHotlabels = convertLabelsOneHot(labels, 10); // ʵ�ֱ�ǩone-hot����
}


//ѵ��SVM�ĺ���
void train_svm(const DataSet& train_set, double C = 10, double gamma = 0.01)
{
    // ת����������vector<vector<float>> -> cv::Mat (N x 784)
    cv::Mat train_features(train_set.feature_vectors.size(), kImageSize, CV_32FC1);
    for (size_t i = 0; i < train_set.feature_vectors.size(); ++i) {
        cv::Mat(1, kImageSize, CV_32FC1, (float*)train_set.feature_vectors[i].data()).copyTo(train_features.row(i));
    }

    // ��ǩ����ת������ʽ����ת����
    cv::Mat train_labels(train_set.labels.size(), 1, CV_32SC1);
    for (size_t i = 0; i < train_set.labels.size(); ++i) {
        train_labels.at<int>(i) = static_cast<int>(train_set.labels[i]);
    }

    // ���������֤
    CV_Assert(train_features.type() == CV_32FC1);
    CV_Assert(train_labels.type() == CV_32SC1);

    // ����SVMģ��
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);       // ���������
    svm->setKernel(cv::ml::SVM::RBF);       // ��˹�ˣ�MNIST�����Կɷ֣�
    svm->setC(C);                          // ���򻯲������轻����֤�Ż���
    svm->setGamma(gamma);                    // RBF�˲�����������������أ�

    // ѵ��ģ�ͣ�Լ��10-30���ӣ�����Ӳ����
    svm->train(train_features, cv::ml::ROW_SAMPLE, train_labels);

    // ����ģ��
    svm->save("mnist_svm.xml");
}

// ���㲢��������ָ��ͻ�������
void save_metrics_and_scores(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    const std::vector<std::vector<float>>& scores, // ÿ��������ÿ��÷�
    const std::string& metrics_file,
    const std::string& scores_file)
{
    int num_classes = kNumClasses;
    std::vector<std::vector<int>> confusion(num_classes, std::vector<int>(num_classes, 0));
    for (size_t i = 0; i < y_true.size(); ++i)
        confusion[y_true[i]][y_pred[i]]++;

    // ����ÿ���precision/recall/f1
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

    // ����ָ��
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

    // ����ÿ����������ʵ��ǩ��Ԥ���ǩ��ÿ��÷֣�����python��ROC/AUC��
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


//SVMģ�Ͳ���
void test_svm(const DataSet& test_set)
{
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("mnist_svm.xml");
    cv::Mat test_features(test_set.feature_vectors.size(), kImageSize, CV_32FC1);
    for (size_t i = 0; i < test_set.feature_vectors.size(); ++i) {
        cv::Mat(1, kImageSize, CV_32FC1, (float*)test_set.feature_vectors[i].data()).copyTo(test_features.row(i));
    }

    // Ԥ���ǩ
    cv::Mat predictions;
    svm->predict(test_features, predictions);

    // ��ȡÿ��������ÿ�����ľ��߷�����one-vs-rest��
    std::vector<std::vector<float>> all_scores(test_set.feature_vectors.size(), std::vector<float>(kNumClasses, 0.0f));
    for (int c = 0; c < kNumClasses; ++c) {
        cv::Ptr<cv::ml::SVM> svm_c = cv::ml::SVM::load("mnist_svm.xml");
        svm_c->setClassWeights(cv::Mat()); // ȷ������Ȩ��
        // ����OpenCV�����SVM��ֱ��֧�ָ��ʣ������þ���ֵ����
        cv::Mat dec_values;
        svm_c->predict(test_features, dec_values, cv::ml::StatModel::RAW_OUTPUT);
        for (int i = 0; i < dec_values.rows; ++i)
            all_scores[i][c] = dec_values.at<float>(i);
    }

    // �ռ���ʵ��ǩ��Ԥ���ǩ
    std::vector<int> y_true, y_pred;
    for (int i = 0; i < predictions.rows; ++i) {
        y_true.push_back(static_cast<int>(test_set.labels[i]));
        y_pred.push_back(static_cast<int>(predictions.at<float>(i)));
    }

    // ����ָ��ͷ���
    save_metrics_and_scores(y_true, y_pred, all_scores, "metrics.csv", "scores.csv");

    // ���׼ȷ��
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
        if (y_true[i] == y_pred[i]) ++correct;
    std::cout << "SVM Accuracy: " << (correct * 100.0 / y_true.size()) << "%\n";
}

// ������֤����������ƽ��׼ȷ��
double cross_validate_svm(const DataSet& dataset, double C, double gamma, int k_folds = 5) {
    // ���������ͱ�ǩ����
    int N = dataset.feature_vectors.size();
    cv::Mat features(N, kImageSize, CV_32FC1);
    for (int i = 0; i < N; ++i)
        cv::Mat(1, kImageSize, CV_32FC1, (float*)dataset.feature_vectors[i].data()).copyTo(features.row(i));
    cv::Mat labels(N, 1, CV_32SC1);
    for (int i = 0; i < N; ++i)
        labels.at<int>(i) = static_cast<int>(dataset.labels[i]);

    // ��������
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42); // �̶����ӱ�֤�ɸ���
    std::shuffle(indices.begin(), indices.end(), rng);

    int fold_size = N / k_folds;
    double total_acc = 0.0;

    for (int fold = 0; fold < k_folds; ++fold) {
        // ����ѵ������֤����
        std::vector<int> val_idx(indices.begin() + fold * fold_size,
            (fold == k_folds - 1) ? indices.end() : indices.begin() + (fold + 1) * fold_size);
        std::vector<int> train_idx;
        train_idx.reserve(N - val_idx.size());
        for (int i : indices) {
            if (std::find(val_idx.begin(), val_idx.end(), i) == val_idx.end())
                train_idx.push_back(i);
        }

        // ����ѵ������֤��
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

        // ѵ��SVM
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setC(C);
        svm->setGamma(gamma);
        svm->train(train_feat, cv::ml::ROW_SAMPLE, train_lab);

        // ��֤
        cv::Mat pred;
        svm->predict(val_feat, pred);
        int correct = 0;
        for (int i = 0; i < pred.rows; ++i) {
            if (static_cast<int>(pred.at<float>(i)) == val_lab.at<int>(i))
                ++correct;
        }
        total_acc += correct * 1.0 / val_idx.size();

        // ��for (int fold = 0; fold < k_folds; ++fold) {...} �ڲ���
        std::cout << "Fold " << fold << " acc: " << (correct * 1.0 / val_idx.size()) << std::endl;

    }
    return total_acc / k_folds;
}

int main()
{
    
    //1.���ݼ�����
    const std::string data_dir = get_data_dir(); // ��ȡ���ݼ�Ŀ¼
    DataSet train_set(data_dir + "/train-images.idx3-ubyte", data_dir + "/train-labels.idx1-ubyte"); // ����ѵ�����ݼ�����
    DataSet test_set(data_dir + "/t10k-images.idx3-ubyte", data_dir + "/t10k-labels.idx1-ubyte"); // �����������ݼ�����

    //2.���ݼ�Ԥ����
    //��һ����ת����ǩ   �ڴ������ݼ�ʱ��ʵ�ֹ�һ����one-hot�����ǩ
    

    //3.ģ�͵���
    ////ֻ�ò�����������������
    //int subset_size = 2000;
    //DataSet small_train_set = get_subset(train_set, subset_size);

    //std::vector<double> C_list = { 0.1, 1, 10, 100 };
    //std::vector<double> gamma_list = { 0.001, 0.01, 0.1, 1 };
    //double best_acc = 0;
    //double best_C = 1, best_gamma = 0.01;
    //for (double C : C_list) {
    //    for (double gamma : gamma_list) {
    //        double acc = cross_validate_svm(small_train_set, C, gamma, 3); // 3�۸���
    //        std::cout << "C=" << C << ", gamma=" << gamma << ", CV acc=" << acc << std::endl;
    //        if (acc > best_acc) {
    //            best_acc = acc;
    //            best_C = C;
    //            best_gamma = gamma;
    //        }
    //    }
    //}
    //std::cout << "Best C=" << best_C << ", Best gamma=" << best_gamma << ", Best CV acc=" << best_acc << std::endl;

    // �����Ų���ѵ��ȫ��ģ��
    //train_svm(train_set, best_C, best_gamma);
    //std::cout << "SVMѵ����ɣ�" << std::endl;

    test_svm(test_set);
    return 0;
}
