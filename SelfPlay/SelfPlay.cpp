#include <iostream>
#include <vector>
#include "Alphapig.cpp"
#include <omp.h>

std::string weightfile = "network.weights";

struct TrainData
{
	float input[9 * 9 * 6] = { 0 };
	float output_policy[9 * 9] = { 0 };
	float output_value = 0;
};

std::vector<TrainData> trainDatas;

static omp_lock_t lock;

TrainData makeTrainData(int* board, int p, std::vector<float> probs, int last_p, bool first)
{
	AlphaPig alphaPig(NULL);
	TrainData res;
	for (int i = 0; i < 81; i++)
	{
		if (board[i] == AI)
			res.input[0 * 81 + i] = 1; // 第一通道为我方棋子
		if (board[i] == OP)
			res.input[1 * 81 + i] = 1; // 第二通道为对方棋子
		if (alphaPig.judgeAvailable(board, i, AI))
			res.input[2 * 81 + i] = 1; // 第三通道为我方允许落子位置
		if (alphaPig.judgeAvailable(board, i, OP))
			res.input[3 * 81 + i] = 1; // 第四通道为对方允许落子位置
		if (first)
			res.input[4 * 81 + i] = 1; // 第五通道为我方是否先手
		res.output_policy[i] = probs[i];
	}
	if (last_p >= 0 && last_p < 81)
		res.input[5 * 81 + last_p] = 1; // 第六通道为对方最后一手位置
	return res;
}

int playGame(Network* net)
{
	srand(time(0) + omp_get_thread_num());
	AlphaPig alphaPig1(net);
	AlphaPig alphaPig2(net);
	std::vector<TrainData> trainData1;
	std::vector<TrainData> trainData2;
	int win = 0;
	int last_p = -1;
	while (true)
	{
		auto a1 = alphaPig1.trainChoose(last_p);
		auto p1 = std::get<0>(a1);
		auto& probs1 = std::get<1>(a1);
		if (p1 == -1)
		{
			win = 2; // a2赢了
			for (int i = 0; i < trainData1.size(); i++)
				trainData1[i].output_value = -1;
			for (int i = 0; i < trainData2.size(); i++)
				trainData2[i].output_value = 1;
			break;
		}
		else
		{
			trainData1.push_back(makeTrainData(alphaPig1.board, p1, probs1, last_p, true));
		}
		alphaPig1.board[p1] = AI;
		alphaPig2.board[p1] = OP;
		last_p = p1;

		auto a2 = alphaPig2.trainChoose(last_p);
		auto p2 = std::get<0>(a2);
		auto& probs2 = std::get<1>(a2);
		if (p2 == -1)
		{
			win = 1; // a1赢了
			for (int i = 0; i < trainData1.size(); i++)
				trainData1[i].output_value = 1;
			for (int i = 0; i < trainData2.size(); i++)
				trainData2[i].output_value = -1;
			break;
		}
		else
		{
			trainData2.push_back(makeTrainData(alphaPig2.board, p2, probs2, last_p, false));
		}
		alphaPig1.board[p2] = OP;
		alphaPig2.board[p2] = AI;
		last_p = p2;
	}
	omp_set_lock(&lock);
	trainDatas.insert(trainDatas.end(), trainData1.begin(), trainData1.end());
	trainDatas.insert(trainDatas.end(), trainData2.begin(), trainData2.end());
	omp_unset_lock(&lock);
	return win;
}

void writeTrainDataToFile(std::string file)
{
	std::ofstream outfile;
	outfile.open(file);
	for (const auto& item : trainDatas)
	{
		for (int i = 0; i < 6 * 81; i++)
			outfile << item.input[i] << " ";
		for (int i = 0; i < 81; i++)
			outfile << item.output_policy[i] << " ";
		outfile << item.output_value << std::endl;
	}
	outfile.close();
}

int main()
{
	system("chcp 65001");
	omp_init_lock(&lock);
	std::cout << "开始生成训练数据" << std::endl;
	auto startTime = time(0);
#pragma omp parallel for
	for (int i = 0; i < 10000; i++)
	{
		Network net(weightfile);
		playGame(&net);
	}
	auto costTime = time(0) - startTime;
	writeTrainDataToFile("dataset/" + std::to_string(time(0)) + ".txt");
	std::cout << "完成生成训练数据，耗时：" << costTime << "s，速度：" << trainDatas.size() / costTime << "步/s" << std::endl;
}
