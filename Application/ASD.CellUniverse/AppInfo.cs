using System.Reflection;

namespace ASD.CellUniverse {

    internal struct AppInfo {

        public static readonly string Name;
        public static readonly string ProcessorArchitecture;
        public static readonly string MajorVersion;
        public static readonly string MinorVersion;
        public static readonly string Build;
        public static readonly string Revision;

        static AppInfo() {

            var assembly = Assembly.GetExecutingAssembly();
            var name = assembly.GetName();
            var version = name.Version;

            Name = assembly.GetCustomAttribute<AssemblyTitleAttribute>().Title;

            ProcessorArchitecture = name.ProcessorArchitecture.ToReadable();

            MajorVersion = version.Major.ToString();
            MinorVersion = version.Minor.ToString();
            Build = version.Build.ToString();
            Revision = version.Revision.ToString();
        }

        public static string ToString(string format = null) {
            if (string.IsNullOrWhiteSpace(format)) {
                return $"{Name} ({ProcessorArchitecture}) - v.{MajorVersion}.{MinorVersion}, build {Build}, rev. {Revision}";
            }
            return string.Format(format, Name, ProcessorArchitecture, MajorVersion, MinorVersion, Build, Revision);
        }
    }

    internal static partial class Extensions {

        public static string ToReadable(this ProcessorArchitecture arch) {

            switch (arch) {
                case ProcessorArchitecture.MSIL: return "Any CPU";
                case ProcessorArchitecture.Amd64: return "x64";
                case ProcessorArchitecture.X86: return "x86";
                case ProcessorArchitecture.Arm: return "ARM";
                case ProcessorArchitecture.IA64: return "IA64";
                default: return "Unspecified";
            }
        }
    }
}