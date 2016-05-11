import subprocess

datasets = ['shaky', 'SANY0025']

for data in datasets:
    print('Running '+data+'...')
    subprocess.check_call(['./build/SubspaceStab', '../data/{}.mp4'.format(data), '--output=build/result_{}'.format(data), '--resize=false', '--crop=false'])
    subprocess.check_call(['ffmpeg', '-i', 'build/result_{}%05d.jpg'.format(data), '-vcodec', 'h264', '-qp', '0', 'result_{}.mp4'.format(data)])
    subprocess.call('rm build/*.jpg', shell=True)
