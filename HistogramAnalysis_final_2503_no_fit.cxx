#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <algorithm>  // for std::replace
// #include <tuple>  // for std::tuple
#include <filesystem> // C++17 or later for creating directories
namespace fs = std::filesystem;

// OPERATIONAL PARAMETERS
const unsigned int Dimension = 3;
typedef signed short PixelType;
typedef unsigned char MaskPixelType;
typedef itk::Image<PixelType, Dimension> ImageType;
typedef itk::Image<MaskPixelType, Dimension> MaskImageType;


// Helper function to remove file extension
std::string removeFileExtension(const std::string& filename) {
    size_t lastDot = filename.find_last_of(".");
    if (lastDot == std::string::npos) { 
		// No dot found, return the original string
        return filename;
    }
    return filename.substr(0, lastDot);
}

// Helper function to split strings
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Helper function to replace substrings in a string
std::string replaceSubstring(const std::string& str, const std::string& oldSub, const std::string& newSub) {
    std::string result = str;
    size_t pos = result.find(oldSub);
    if (pos != std::string::npos) {
        result.replace(pos, oldSub.length(), newSub);
    }
    return result;
}

// Helper function to get parent directory
std::string getParentDirectory(const std::string& filepath) {
    size_t found = filepath.find_last_of("/\\");  // Works for both Unix and Windows paths
    if (found != std::string::npos) {
        return filepath.substr(0, found);
    }
    return filepath;  // If no slash found, return original
}


// Function to combine bduct and pduct histograms
std::map<int, unsigned int> combineHistograms(
    const std::map<int, unsigned int>& hist1,
    const std::map<int, unsigned int>& hist2) {
    
    std::map<int, unsigned int> combined_hist = hist1;
    for (const auto& entry : hist2) {
        combined_hist[entry.first] += entry.second;
    }
    return combined_hist;
}

// Function to calculate mean
double calculateMean(const std::map<int, unsigned int>& histogram) {
    double sum = 0.0, count = 0.0;
    for (const auto& entry : histogram) {
        sum += entry.first * entry.second;
        count += entry.second;
    }
    return count > 0 ? sum / count : 0.0;
}

// Function to calculate median
double calculateMedian(const std::map<int, unsigned int>& histogram) {
    std::vector<std::pair<int, unsigned int>> entries(histogram.begin(), histogram.end());
    std::vector<unsigned int> cumulativeFrequency;
    cumulativeFrequency.reserve(entries.size());

    for (size_t i = 0; i < entries.size(); ++i) {
        cumulativeFrequency.push_back(i == 0 ? entries[i].second : cumulativeFrequency[i - 1] + entries[i].second);
    }

    unsigned int totalFrequency = cumulativeFrequency.back();
    unsigned int medianIndex = totalFrequency / 2;

    for (size_t i = 0; i < cumulativeFrequency.size(); ++i) {
        if (cumulativeFrequency[i] >= medianIndex) {
            return entries[i].first;
        }
    }
    return 0.0;
}

// Function to calculate standard deviation
double calculateStdDev(const std::map<int, unsigned int>& histogram, double mean) {
    double sum = 0.0, count = 0.0;
    for (const auto& entry : histogram) {
        sum += entry.second * (entry.first - mean) * (entry.first - mean);
        count += entry.second;
    }
    return count > 0 ? std::sqrt(sum / count) : 0.0;
}

// Function to save statistics to a file
// void saveMaskStatisticsToCSV(const std::map<std::string, std::tuple<double, double, double>>& stats, const std::string& filename) {
void saveMaskStatisticsToCSV(const std::map<std::string, std::tuple<double, double, double, double>>& stats, const std::string& filename) {
    std::ofstream statsFile(filename);
    if (!statsFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
        return;
    }

    // Write header
    statsFile << "Mask,Mean,Median,StdDev" << std::endl;

    // Write rows for each mask type in the correct order
    statsFile << "bduct,"
              << std::get<0>(stats.at("mean")) << ","
              << std::get<0>(stats.at("median")) << ","
              << std::get<0>(stats.at("std")) << std::endl;

    statsFile << "pduct,"
              << std::get<1>(stats.at("mean")) << ","
              << std::get<1>(stats.at("median")) << ","
              << std::get<1>(stats.at("std")) << std::endl;

    statsFile << "pancreas,"
              << std::get<2>(stats.at("mean")) << ","
              << std::get<2>(stats.at("median")) << ","
              << std::get<2>(stats.at("std")) << std::endl;

    // combined statistics (buct + pduct)
    statsFile << "combinedDuct,"
              << std::get<3>(stats.at("mean")) << ","
              << std::get<3>(stats.at("median")) << ","
              << std::get<3>(stats.at("std")) << std::endl;      

    statsFile.close();
    std::cout << "\033[36m-- Saved statistics to: " << filename << "\033[0m"<< std::endl;
}


// Save a histogram to a CSV file
// * Only count non-zero frequency intensity values (exclude intensity with 0 frequency)
void saveHistogramToCSV(const std::map<int, unsigned int>& histogram, const std::string& filename) {
    std::ofstream csv_file(filename);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
        return;
    }
    csv_file << "Intensity,Frequency" << std::endl;
    for (const auto& entry : histogram) {
        csv_file << entry.first << "," << entry.second << std::endl;
    }
    csv_file.close();
    std::cout << "\033[36m-- Saved histogram to: " << filename << "\033[0m"<< std::endl;
}


// Function to Read Threshold CSV
std::map<std::string, double> readThresholdsFromCSV(const std::string& filename) {
    std::map<std::string, double> thresholds;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open threshold file: " << filename << std::endl;
        return thresholds;
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string key;
        double value;
        
        if (std::getline(ss, key, ',') && (ss >> value)) {
            thresholds[key] = value;
        } else {
            std::cerr << "Warning: Skipping malformed line: " << line << std::endl;
        }
    }
    file.close();
    return thresholds;
}


// Save image data to a file
void MakeImageFile(ImageType::Pointer image, const char* filename) {
    typedef itk::ImageFileWriter<ImageType> WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetInput(image);
    writer->SetFileName(filename);
    writer->Update();
}

// Save mask data to a file
void MakeMaskImageFile(MaskImageType::Pointer image, const char* filename) {
    typedef itk::ImageFileWriter<MaskImageType> WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetInput(image);
    writer->SetFileName(filename);
    writer->Update();
}


#include "itkImageIOFactory.h"
#include "itkNiftiImageIOFactory.h"
#include "itkPNGImageIOFactory.h"
#include "itkTIFFImageIOFactory.h"

int main(int argc, char* argv[]) {
	std::cout << "\n** Custom version 2503 - COMBINE DUCT HISTOGRAMS + NO FITTING ** " << std::endl; 

    // static bool isRegistered = false;
    // if (!isRegistered) {
    //     itk::NiftiImageIOFactory::RegisterOneFactory();  // Register NIfTI IO Factory only once
    //     isRegistered = true;
    // }
    itk::NiftiImageIOFactory::RegisterOneFactory();
    itk::PNGImageIOFactory::RegisterOneFactory();
    itk::TIFFImageIOFactory::RegisterOneFactory();
    std::cout << "** (ITK IO Factories Registered Successfully) **" << std::endl;

	/* set and check parameters */ 
	std::cout << "Arguments passed: " << std::endl; 
    for (int i = 0; i < argc; ++i) {
        std::cout << "arg[" << i << "] = " << argv[i] << std::endl;
    }

    if (argc < 9) {
		std::cerr << "Missing Parameters " << std::endl;
        std::cerr << "Usage: " << argv[0]
                  << " FID   imageFile	pancreasMaskFile    bductMaskFile   pductMaskFile   scaledFile	thresFile   CSVFile" << std::endl;
        std::cerr << std::endl;
		return EXIT_FAILURE;
    }

    std::cout << "Input file ID: " << argv[1] << std::endl;
	std::cout << "Input image file: " << argv[2] << std::endl;
    std::cout << "Input mask file (pancreas): " << argv[3] << std::endl;
    std::cout << "Input mask file (bduct): " << argv[4] << std::endl;
    std::cout << "Input mask file (pduct): " << argv[5] << std::endl;

    std::string FID = argv[1]; // FID
    std::string imageFile = argv[2]; // imageFile
    std::string pancreasMaskFile = argv[3]; // pancreasMaskFile
    std::string bductMaskFile = argv[4]; // bductMaskFile
    std::string pductMaskFile = argv[5]; // pductMaskFile
    std::string scaledFile = argv[6]; // scaledFile
    std::string thresFile = argv[7]; // thresFile
    std::string CSVFile = argv[8]; // CSVFile

    typedef itk::ImageFileReader<ImageType> ImageReaderType;
    ImageReaderType::Pointer imageReader = ImageReaderType::New();
    imageReader->SetFileName(imageFile);

    typedef itk::ImageFileReader<MaskImageType> MaskReaderType;
    MaskReaderType::Pointer pancreasMaskReader = MaskReaderType::New();
    pancreasMaskReader->SetFileName(pancreasMaskFile);

    typedef itk::ImageFileReader<MaskImageType> MaskReaderType;
    MaskReaderType::Pointer bductMaskReader = MaskReaderType::New();
    bductMaskReader->SetFileName(bductMaskFile);

    typedef itk::ImageFileReader<MaskImageType> MaskReaderType;
    MaskReaderType::Pointer pductMaskReader = MaskReaderType::New();
    pductMaskReader->SetFileName(pductMaskFile);

    try {
        imageReader->Update();
        pancreasMaskReader->Update();
        bductMaskReader->Update();
        pductMaskReader->Update();
		std::cout << "Input files successfully read ! " << std::endl;
    } catch (itk::ExceptionObject& ex) {
		std::cout << ex << std::endl;
        std::cerr << "Error reading files: " << ex << std::endl;
        return EXIT_FAILURE;
    }

    ImageType::Pointer image = imageReader->GetOutput();
    MaskImageType::Pointer pancreasMask = pancreasMaskReader->GetOutput();
    MaskImageType::Pointer bductMask = bductMaskReader->GetOutput();
    MaskImageType::Pointer pductMask = pductMaskReader->GetOutput();

	ImageType::SizeType size = image->GetBufferedRegion().GetSize();
    PixelType* lpData = image->GetBufferPointer();
    MaskPixelType* lpPancreasMaskData = pancreasMask->GetBufferPointer();
    MaskPixelType* lpBDuctMaskData = bductMask->GetBufferPointer();
    MaskPixelType* lpPDuctMaskData = pductMask->GetBufferPointer();


    // Compute histogram
    std::map<int, unsigned int> pancreasHistogram, bductHistogram, pductHistogram, combinedHistogram;
    int min = 1000000, max = -1000000;
    for (int z = 0; z < size[2]; z++) {
        for (int y = 0; y < size[1]; y++) {
            for (int x = 0; x < size[0]; x++) {
                int index = z * size[1] * size[0] + y * size[0] + x;
                // int intensity = lpData[index];
                if (lpPancreasMaskData[index] > 0) {
                    pancreasHistogram[lpData[index]]++;
                    if (lpData[index] < min) min = lpData[index];
                    if (lpData[index] > max) max = lpData[index];
                }
                if (lpBDuctMaskData[index] > 0) {
                    bductHistogram[lpData[index]]++;
                    combinedHistogram[lpData[index]]++;  // Add bduct to combined histogram
                }
                if (lpPDuctMaskData[index] > 0) {
                    pductHistogram[lpData[index]]++;
                    combinedHistogram[lpData[index]]++;  // Add pduct to combined histogram
                }
            }
        }
    }

    // Save each histogram to a CSV file
    // if (pancreasHistogram.empty() || bductHistogram.empty() || pductHistogram.empty()) {
    //     std::cerr << "[ERROR] One or more histograms are empty. Exiting." << std::endl;
    //     return EXIT_FAILURE;
    // }
    // // works for case with empty mask
    if (pancreasHistogram.empty()) {
        std::cerr << "[WARNING] Pancreas histogram is empty!" << std::endl;
    }
    if (bductHistogram.empty()) {
        std::cerr << "[WARNING] Bduct histogram is empty!" << std::endl;
    }
    if (pductHistogram.empty()) {
        std::cerr << "[WARNING] Pduct histogram is empty!" << std::endl;
    }
    if (pancreasHistogram.empty() && bductHistogram.empty() && pductHistogram.empty()) {
        std::cerr << "[ERROR] All histograms are empty. Nothing to process." << std::endl;
        return EXIT_FAILURE;
    }

    std::string bductCSVFile = replaceSubstring(CSVFile, "_pancreas", "_bduct");
    std::string pductCSVFile = replaceSubstring(CSVFile, "_pancreas", "_pduct");
    std::string combinedCSVFile = replaceSubstring(CSVFile, "_pancreas", "_combinedDuct");
    // saveHistogramToCSV(bductHistogram, bductCSVFile);
    // saveHistogramToCSV(pductHistogram, pductCSVFile);
    // saveHistogramToCSV(pancreasHistogram, CSVFile);
    // saveHistogramToCSV(combinedHistogram, combinedCSVFile);
    // // works for case with empty mask
    if (!bductHistogram.empty()) {
        saveHistogramToCSV(bductHistogram, bductCSVFile);
    }
    if (!pductHistogram.empty()) {
        saveHistogramToCSV(pductHistogram, pductCSVFile);
    }
    if (!pancreasHistogram.empty()) {
        saveHistogramToCSV(pancreasHistogram, CSVFile);
    }
    if (!combinedHistogram.empty()) {
        saveHistogramToCSV(combinedHistogram, combinedCSVFile);
    }
    

    // Calculate statistics for each mask
    std::map<std::string, std::tuple<double, double, double, double>> metricStatistics;
    // metricStatistics["mean"] = {
    //     calculateMean(bductHistogram),
    //     calculateMean(pductHistogram),
    //     calculateMean(pancreasHistogram),
    //     calculateMean(combinedHistogram)  // combined
    // };
    // metricStatistics["median"] = {
    //     calculateMedian(bductHistogram),
    //     calculateMedian(pductHistogram),
    //     calculateMedian(pancreasHistogram),
    //     calculateMedian(combinedHistogram)  // combined
    // };
    // metricStatistics["std"] = {
    //     calculateStdDev(bductHistogram, std::get<0>(metricStatistics["mean"])),
    //     calculateStdDev(pductHistogram, std::get<1>(metricStatistics["mean"])),
    //     calculateStdDev(pancreasHistogram, std::get<2>(metricStatistics["mean"])),
    //     calculateStdDev(combinedHistogram, std::get<3>(metricStatistics["mean"]))  // combined
    // };
    // // works for case with empty mask
    auto safe_stat = [](const std::map<int, unsigned int>& hist, double mean = 0) {
        return std::make_tuple(
            hist.empty() ? 0.0 : calculateMean(hist),
            hist.empty() ? 0.0 : calculateMedian(hist),
            hist.empty() ? 0.0 : calculateStdDev(hist, mean),
            hist.empty() ? 0.0 : static_cast<double>(hist.size())
        );
    };

    auto b_stats = safe_stat(bductHistogram);
    auto p_stats = safe_stat(pductHistogram);
    auto pan_stats = safe_stat(pancreasHistogram);
    auto comb_stats = safe_stat(combinedHistogram);

    metricStatistics["mean"] = {
        std::get<0>(b_stats),
        std::get<0>(p_stats),
        std::get<0>(pan_stats),
        std::get<0>(comb_stats)
    };
    metricStatistics["median"] = {
        std::get<1>(b_stats),
        std::get<1>(p_stats),
        std::get<1>(pan_stats),
        std::get<1>(comb_stats)
    };
    metricStatistics["std"] = {
        std::get<2>(b_stats),
        std::get<2>(p_stats),
        std::get<2>(pan_stats),
        std::get<2>(comb_stats)
    };
    std::string statsCSVFile = replaceSubstring(CSVFile, "_pancreas", "_all_mask");
    statsCSVFile = replaceSubstring(statsCSVFile, "_histogram", "_histogram_statistics");  // "_Histogram"
    saveMaskStatisticsToCSV(metricStatistics, statsCSVFile);



    // Generate pure histogram
    // works for case with empty mask
    if (fs::exists(bductCSVFile)) {
        std::string bductCommand = std::string("python -u histogram_thresh.py") +
                                " --file_id " + FID +
                                " --file_paths " + bductCSVFile +
                                " --means " + std::to_string(std::get<0>(metricStatistics["mean"])) +
                                " --medians " + std::to_string(std::get<0>(metricStatistics["median"])) +
                                " --stds " + std::to_string(std::get<0>(metricStatistics["std"]));
        std::cout << "\nRunning thresholding with Histogram: " << bductCommand << std::endl;
        system(bductCommand.c_str());
    } else {
        std::cerr << "[INFO] Skipping bductCommand because histogram file not found: " << bductCSVFile << std::endl;
    }
    if (fs::exists(pductCSVFile)) {
        std::string pductCommand = std::string("python -u histogram_thresh.py") +
                                " --file_id " + FID +
                                " --file_paths " + pductCSVFile +
                                " --means " + std::to_string(std::get<1>(metricStatistics["mean"])) +
                                " --medians " + std::to_string(std::get<1>(metricStatistics["median"])) +
                                " --stds " + std::to_string(std::get<1>(metricStatistics["std"]));
        std::cout << "\nRunning thresholding with Histogram: " << pductCommand << std::endl;
        system(pductCommand.c_str());
    } else {
        std::cerr << "[INFO] Skipping pductCommand because histogram file not found: " << pductCSVFile << std::endl;
    }
    if (fs::exists(CSVFile)) {
        std::string pancreasCommand = std::string("python -u histogram_thresh.py") +
                                    " --file_id " + FID +
                                    " --file_paths " + CSVFile +
                                    " --means " + std::to_string(std::get<2>(metricStatistics["mean"])) +
                                    " --medians " + std::to_string(std::get<2>(metricStatistics["median"])) +
                                    " --stds " + std::to_string(std::get<2>(metricStatistics["std"]));
        std::cout << "\nRunning thresholding with Histogram: " << pancreasCommand << std::endl;
        system(pancreasCommand.c_str());
    } else {
        std::cerr << "[INFO] Skipping pancreasCommand because histogram file not found: " << CSVFile << std::endl;
    }
    if (fs::exists(combinedCSVFile)) {
        std::string combinedCommand = std::string("python -u histogram_thresh.py") +
                                    " --file_id " + FID +
                                    " --file_paths " + combinedCSVFile +
                                    " --means " + std::to_string(std::get<3>(metricStatistics["mean"])) +
                                    " --medians " + std::to_string(std::get<3>(metricStatistics["median"])) +
                                    " --stds " + std::to_string(std::get<3>(metricStatistics["std"]));
        std::cout << "\nRunning thresholding with Histogram: " << combinedCommand << std::endl;
        system(combinedCommand.c_str());
    } else {
        std::cerr << "[INFO] Skipping combinedCommand because histogram file not found: " << combinedCSVFile << std::endl;
    }


    // Generate overlapped histogram & threshold
    // std::string allMaskOverlappedCommand = "python -u histogram_thresh.py " + bductCSVFile + " " + pductCSVFile + " " + CSVFile + " " + combinedCSVFile +
    //                                 " --means " + std::to_string(std::get<0>(metricStatistics["mean"])) + " " +
    //                                             std::to_string(std::get<1>(metricStatistics["mean"])) + " " +
    //                                             // std::to_string(std::get<2>(metricStatistics["mean"])) +
    //                                             std::to_string(std::get<2>(metricStatistics["mean"])) + " " +
    //                                             std::to_string(std::get<3>(metricStatistics["mean"])) +
    //                                 " --medians " + std::to_string(std::get<0>(metricStatistics["median"])) + " " +
    //                                             std::to_string(std::get<1>(metricStatistics["median"])) + " " +
    //                                             // std::to_string(std::get<2>(metricStatistics["median"])) +
    //                                             std::to_string(std::get<2>(metricStatistics["median"])) + " " +
    //                                             std::to_string(std::get<3>(metricStatistics["median"])) +
    //                                 " --stds " + std::to_string(std::get<0>(metricStatistics["std"])) + " " +
    //                                             std::to_string(std::get<1>(metricStatistics["std"])) + " " +
    //                                             // std::to_string(std::get<2>(metricStatistics["std"]));
    //                                             std::to_string(std::get<2>(metricStatistics["std"])) + " " +
    //                                             std::to_string(std::get<3>(metricStatistics["std"]));
    // std::cout << "\nThresholding for all masks overlapped histogram: " << allMaskOverlappedCommand << std::endl;
    // system(allMaskOverlappedCommand.c_str());

    // works for case with empty mask
    std::vector<std::string> availableFiles;
    std::vector<std::string> labels;  // corresponding to order: bduct, pduct, pancreas, combinedDuct
    std::vector<double> means, medians, stds;

    if (fs::exists(bductCSVFile)) {
        availableFiles.push_back(bductCSVFile);
        labels.push_back("bduct");
        means.push_back(std::get<0>(metricStatistics["mean"]));
        medians.push_back(std::get<0>(metricStatistics["median"]));
        stds.push_back(std::get<0>(metricStatistics["std"]));
    }
    if (fs::exists(pductCSVFile)) {
        availableFiles.push_back(pductCSVFile);
        labels.push_back("pduct");
        means.push_back(std::get<1>(metricStatistics["mean"]));
        medians.push_back(std::get<1>(metricStatistics["median"]));
        stds.push_back(std::get<1>(metricStatistics["std"]));
    }
    if (fs::exists(CSVFile)) {
        availableFiles.push_back(CSVFile);
        labels.push_back("pancreas");
        means.push_back(std::get<2>(metricStatistics["mean"]));
        medians.push_back(std::get<2>(metricStatistics["median"]));
        stds.push_back(std::get<2>(metricStatistics["std"]));
    }
    if (fs::exists(combinedCSVFile)) {
        availableFiles.push_back(combinedCSVFile);
        labels.push_back("combinedDuct");
        means.push_back(std::get<3>(metricStatistics["mean"]));
        medians.push_back(std::get<3>(metricStatistics["median"]));
        stds.push_back(std::get<3>(metricStatistics["std"]));
    }

    if (!availableFiles.empty()) {
        std::string allMaskOverlappedCommand = std::string("python -u histogram_thresh.py");
        allMaskOverlappedCommand += " --file_id " + FID;
        allMaskOverlappedCommand += " --file_paths";
        for (const auto& f : availableFiles) {
            allMaskOverlappedCommand += " " + f;
        }
        allMaskOverlappedCommand += " --means";
        for (const auto& m : means) {
            allMaskOverlappedCommand += " " + std::to_string(m);
        }
        allMaskOverlappedCommand += " --medians";
        for (const auto& m : medians) {
            allMaskOverlappedCommand += " " + std::to_string(m);
        }
        allMaskOverlappedCommand += " --stds";
        for (const auto& s : stds) {
            allMaskOverlappedCommand += " " + std::to_string(s);
        }

        std::cout << "\nRunning overlapped histogram with available files: " << allMaskOverlappedCommand << std::endl;
        system(allMaskOverlappedCommand.c_str());
    } else {
        std::cerr << "[INFO] No histogram files found for overlap. Skipping." << std::endl;
    }
    

    // std::string base_filename = removeFileExtension(CSVFile);
    // std::cout << "Base filename for CSV files: " << base_filename << std::endl;  // C:\Users\MI2RL\Desktop\01_duct_segmentation\cyst_examples_with_merged\_temp_test_2503\17286516\mask_statistics\17286516_t2_pancreas_histogram
    std::string base_filepath = getParentDirectory(CSVFile);
    std::cout << "\n\033[1m>> Base filepath for statistics files: " << base_filepath << "\033[0m" << std::endl;  // output/37581291/mask_statistics

    // Iterate through all possible CSV files for fitted parameters
    std::vector<std::string> csv_variants = {
        "threshold.csv"
    };

    std::vector<std::string> threshold_keys = {
        // "zscore_5.0", "percentile_30", "percentile_50", "FWHM", "mixture_intersection_dynamic_weight_skew",
        "point_30", "point_40", "point_50", "point_60", "point_70"
    };

    try {
        int processed_files = 0, skipped_files = 0;
        for (const auto& suffix : csv_variants) {
            std::cout << "" << std::endl;

            // std::string csv_thresh_filename = removeFileExtension(CSVFile) + suffix;  // "_gaussian_parameters.csv";
            // std::ifstream csv_fit_file(csv_thresh_filename);
            // std::cout << csv_thresh_filename << std::endl;
            
            std::string csv_thresh_filename = base_filepath + "/" + FID + "_t2_" + suffix;
            std::ifstream csv_fit_file(csv_thresh_filename);

            if (!csv_fit_file.is_open()) {
                std::cout << ">> File not found: " << csv_thresh_filename << std::endl;
                skipped_files++;
                continue;
            }
            std::cout << "\033[1m>> Processing file: " << csv_thresh_filename << "\033[0m" << std::endl;
            processed_files++;


            // Read Thresholds
            std::map<std::string, double> thresholds = readThresholdsFromCSV(csv_thresh_filename);
            if (thresholds.empty()) {
                std::cerr << "Error: No threshold values found in " << csv_thresh_filename << std::endl;
                continue;
            }
            // std::cout << "Thresholds read from CSV : ";
            // for (const auto& pair : thresholds) {
            //     std::cout << "[" << pair.first << "] " << pair.second << std::endl;
            // }


            // Apply thresholding for each threshold type
            for (const auto& key : threshold_keys) {
                if (thresholds.find(key) == thresholds.end()) {
                    // std::cout << "" << std::endl;
                    // std::cerr << "File:" << csv_thresh_filename << std::endl;
                    // std::cerr << "Warning: Missing threshold value for " << key << std::endl;
                    continue;
                }
                std::cout << "" << std::endl;
                
                double thresholdValue = thresholds[key];
                std::cout << ">> Applying Threshold : [" << key << "] " << thresholdValue << std::endl;

                // // Generate output filenames
                std::cout << "\033[1m>> Generated filenames :\033[0m" << std::endl;
                // std::cout <<  "Scaled: " << scaledFile << std::endl;
                // std::cout <<  "Thresholded: " << thresFile << std::endl;

                // Thresholding logic remains the same, but outputs are written to unique files
                ImageType::Pointer scaled = ImageType::New();
                scaled->SetRegions(image->GetBufferedRegion());
                scaled->CopyInformation(image);
                scaled->Allocate();
                scaled->FillBuffer(static_cast<PixelType>(0));

                MaskImageType::Pointer thres = MaskImageType::New();
                thres->SetRegions(image->GetBufferedRegion());
                thres->CopyInformation(image);
                thres->Allocate();
                thres->FillBuffer(static_cast<MaskPixelType>(0));

                PixelType* lpScaledData = scaled->GetBufferPointer();
                MaskPixelType* lpThresData = thres->GetBufferPointer();


                // double upper_threshold = fitted_mean + z_threshold * fitted_std;
                // if (upper_threshold > max_val) {
                //     upper_threshold = max_val;
                // }
                for (int z = 0; z < size[2]; z++) {
                    for (int y = 0; y < size[1]; y++) {
                        for (int x = 0; x < size[0]; x++) {
                            int index = z * size[1] * size[0] + y * size[0] + x;
                            if (lpPancreasMaskData[index] > 0) {
                                // lpScaledData[index] = static_cast<PixelType>(
                                //     1000.0 * (lpData[index] - fitted_mean) / fitted_std); // z-score normalized intensity
                                // if (lpData[index] > upper_threshold) {
                                //     lpThresData[index] = 2; // Cyst
                                // } else {
                                //     lpThresData[index] = 1; // Non-cyst
                                // }
                                lpScaledData[index] = static_cast<PixelType>(lpData[index]); // raw intensity
                                lpThresData[index] = (lpData[index] > thresholdValue) ? 2 : 1;
                            }
                        }
                    }
                }
                
                // Save the scaled and thresholded images for the current CSV variant
                // MakeImageFile(scaled, scaledFile.c_str());
                try {
                    MakeImageFile(scaled, scaledFile.c_str());
                    std::cout << "\033[36m-- Scaled image    : " << scaledFile << "\033[0m" << std::endl;
                } catch (const std::exception& ex) {
                    std::cerr << "\033[1m-- Error saving scaled image: " << ex.what() << "\033[0m" << std::endl;
                }
                // MakeMaskImageFile(thres, thresFile.c_str());
                try {
                    MakeMaskImageFile(thres, thresFile.c_str());
                    std::cout << "\033[36m-- Thresholded mask : " << thresFile << "\033[0m"  << std::endl;
                } catch (const std::exception& ex) {
                    std::cerr << "\033[1m-- Error saving thresholded mask: " << ex.what() << "\033[0m" << std::endl;
                }
            }
        }
        std::cout << "\n" << std::endl;
        std::cout << "Total processed files: " << processed_files << std::endl;
        std::cout << "Total skipped files: " << skipped_files << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "Exception caught during loop: " << ex.what() << std::endl;
    }

    return EXIT_SUCCESS;
}