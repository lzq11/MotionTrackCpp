#pragma once
#include <glob.h>
#include <vector>
#include <string>
#include <sys/stat.h>

// using namespace std;

#include <dirent.h>
#include <stdio.h>
void read_dir(std::string fileDir, std::vector<std::string>& files)
{
    struct dirent *ent = NULL;
    DIR *dir = opendir(fileDir.c_str());
    if (dir != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0)
            {
                //std::cout << ent->d_name << std::endl;
                files.push_back(ent->d_name);
            }
        }
        closedir(dir);

    }
    else
    {
        std::cout << "failed to open directory" << std::endl;
    }
}



int judge(string path)
{
    struct stat s;
    if (stat(path.c_str(), &s) == 0)
    {
        if(s.st_mode & S_IFDIR)
            return 1;
        else if(s.st_mode & S_IFREG)
            return 2;
        else
            return -1;
    }
    return -1;
}

vector<string> globVector(string path)
{
    const string pattern =  path + "/*";
    glob_t glob_result;
    glob(pattern.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        files.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}

void GetFiles(string dir, vector<string> &files)
{
    int mk = judge(dir);
    if(mk == -1)
        return;
    else if(mk == 2)
    {
        files.push_back(dir);
        return;
    }
    vector<string> entry = globVector(dir);
    for(auto iter = entry.begin(); iter != entry.end(); iter++)
    {
        mk = judge(*iter);
        if (mk == -1)
            continue;
        else if(mk == 1)
            GetFiles(*iter, files);
        else if(mk == 2)
            files.push_back(*iter);
    }
    vector<string>().swap(entry);
}

struct Result
{
    int frame_id;
    vector<int> online_ids;
    vector<vector<float>> tlwhs;
    vector<float> online_scores;
    vector<int> online_classids;
};


void write_results_with_socre_and_cls(string save_path,vector<Result> results )
{
    ofstream outfile;
    outfile.open(save_path,ios::out);

    for (Result result:results)
	{
        //fixed << setprecision(2)是为了保留小数点后2位进行写入
        for (int i=0; i != result.online_scores.size();++i)
        {
            outfile << result.frame_id <<",";
		    outfile << result.online_ids[i]<<",";
            // outfile << fixed << setprecision(0)<< result.tlwhs[i][0]<<",";
            // outfile << fixed << setprecision(0)<< result.tlwhs[i][1]<<",";
            // outfile << fixed << setprecision(0)<< result.tlwhs[i][2]<<",";
            // outfile << fixed << setprecision(0)<< result.tlwhs[i][3]<<",";
            outfile << int(result.tlwhs[i][0])<<",";
            outfile << int(result.tlwhs[i][1])<<",";
            outfile << int(result.tlwhs[i][2])<<",";
            outfile << int(result.tlwhs[i][3])<<",";
            outfile << "1"<<",";
		    outfile << result.online_classids[i] + 1 <<",";
            outfile << "0"<<"\n";
        }
	}
	outfile.close();//关闭文件，保存文件。
}