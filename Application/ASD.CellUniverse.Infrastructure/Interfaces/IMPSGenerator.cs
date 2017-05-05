using System;
using System.Windows.Media;

namespace ASD.CellUniverse.Infrastructure.Interfaces {

    internal interface IMPSGenerator {

        DoubleCollection MPSCollection { get; }

        double MPS { get; set; }

        event Action NextFrameTime;
        void Start();
        void Stop();
    }
}