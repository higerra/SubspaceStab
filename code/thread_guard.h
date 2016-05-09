#ifndef THREAD_GUARD
#define THREAD_GUARD
#include <thread>
#include <mutex>
class thread_guard{
public:
	thread_guard():t(std::thread()){};
	explicit thread_guard(std::thread& t_){
		bind(t_);
	}

	void bind(std::thread& t_){
		t = std::move(t_);
	}
	void join(){
		if(t.joinable())
			t.join();
	}

	thread_guard(thread_guard const&) = delete;
	thread_guard& operator= (thread_guard const&) = delete;
	~thread_guard(){
		if(t.joinable())
			t.join();
	}
private:
	std::thread t;
};
#endif












