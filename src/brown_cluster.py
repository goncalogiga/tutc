import subprocess

class BrownCluster():
    def __init__(self, text, chk=False, stats=False, paths2map=False,
                 no_prune=False, ncollocs=500, c=1000, plen=1, min_occur=1,
                 rand=-850248136, threads=1, output_dir=None, restrict=None,
                 paths=None, _map=None, collocs=None, featvec=None,
                 comment=None):

        args = ["--chk", "--stats", "--paths2map", "--no_prune", "--ncollocs",
                "--c", "--plen", "--min-occur", "--rand", "--threads",
                "--output_dir", "--restrict", "--paths", "--map", "--collocs",
                "--featvec"]

        selected_args = []

        if not isinstance(chk, bool):
            raise Exception("chk should be either True or False")
        if chk is True:
            selected_args.append("")
        else:
            selected_args.append(None)

        if not isinstance(stats, bool):
            raise Exception("chk should be either True or False")
        if stats is True:
            selected_args.append("")
        else:
            selected_args.append(None)

        if not isinstance(paths2map, bool):
            raise Exception("chk should be either True or False")
        if paths2map is True:
            selected_args[2].append("")
        else:
            selected_args[2].append(None)

        if not isinstance(no_prune, bool):
            raise Exception("chk should be either True or False")
        if no_prune is True:
            selected_args[3].append("")
        else:
            selected_args[3].append(None)

        selected_args.append(ncollocs)
        selected_args.append(c)
        selected_args.append(plen)
        selected_args.append(min_occur)
        selected_args.append(rand)
        selected_args.append(threads)
        selected_args.append(output_dir)
        selected_args.append(restrict)
        selected_args.append(paths)
        selected_args.append(_map)
        selected_args.append(collocs)
        selected_args.append(featvec)

        self.cmd = "./lib/brown_cluster/wcluser "

        for arg, value in zip(args, selected_args):
            if value is not None:
                self.cmd += arg + " " + value + " "

    def run(self):
        popen = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)
        popen.wait()
        return popen.stdout.read()
