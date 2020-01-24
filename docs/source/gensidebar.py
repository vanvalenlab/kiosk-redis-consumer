# Credit: https://github.com/robotpy/robotpy-docs/blob/master/gensidebar.py
# This file generates the sidebar/toctree for all RobotPy projects and should
# be copied to each project when it is updated
#

import os


def write_sidebar(fname, contents):
    with open(fname, "w") as fp:
        fp.write(contents)


def generate_sidebar(conf, conf_api):

    # determine 'latest' or 'stable'
    # if not conf.do_gen:
    do_gen = os.environ.get("SIDEBAR", None) == "1" or conf["on_rtd"]
    version = conf["rtd_version"]

    lines = ["", ".. DO NOT MODIFY! THIS PAGE IS AUTOGENERATED!", ""]

    def toctree(name):
        lines.extend(
            [".. toctree::", "    :caption: %s" % name, "    :maxdepth: 3", ""]
        )

    def endl():
        lines.append("")

    def write(desc, link):
        if conf_api == "deepcell-kiosk":
            args = desc, link
        elif not do_gen:
            return
        else:
            args = (
                desc,
                "https://deepcell-kiosk.readthedocs.io/en/%s/%s.html" % (version, link),
            )

        lines.append("    %s <%s>" % args)

    def write_api(project, desc):
        if project != conf_api:
            if do_gen:
                args = desc, project, version
                lines.append(
                    "    %s API <https://deepcell-kiosk.readthedocs.io/projects/%s/en/%s/api.html>"
                    % args
                )
        else:
            lines.append("    %s API <api>" % desc)

    def write_subproject(project, desc):
        if project != conf_api:
            if do_gen:
                lines.append(
                    ('    {desc} <https://deepcell-kiosk.readthedocs.io/projects/'
                     '{project}/en/{version}/>').format(
                        desc=desc,
                        project=project,
                        version=version
                    )
                )
        else:
            lines.append('    {desc} <index>'.format(desc=desc))

    #
    # Specify the sidebar contents here
    #

    toctree('Deepcell Kiosk')
    write('Getting Started', 'GETTING_STARTED')
    write('Troubleshooting', 'TROUBLESHOOTING')
    write('Tutorial: Custom Jobs', 'CUSTOM-JOB')
    write('Advanced Documentation', 'ADVANCED_DOCUMENTATION')
    write('Software Infrastructure', 'SOFTWARE_INFRASTRUCTURE')
    write('Developer Documentation', 'DEVELOPER')
    endl()

    toctree('Container Reference')
    write_subproject('kiosk-redis-consumer', 'kiosk-redis-consumer')
    write_subproject('kiosk-frontend', 'kiosk-frontend')
    endl()

    print(lines)

    write_sidebar("_sidebar.rst.inc", "\n".join(lines))
