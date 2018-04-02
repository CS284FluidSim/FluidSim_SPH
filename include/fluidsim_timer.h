#ifndef _FLUIDSIM_TIMER_H_
#define _FLUIDSIM_TIMER_H_


namespace FluidSim {
	class Timer
	{
	private:
		int frames;
		int update_time;
		int last_time;
		double FPS;

	public:
		Timer();
		void update();
		double get_fps();
	};
}

#endif
