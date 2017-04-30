using System;
using System.Windows.Media;

namespace ASD.CellUniverse.Infrastructure.Interfaces {

    internal interface IFPSGenerator {

        DoubleCollection FPSCollection { get; }

        double FPS { get; set; }

        event Action NextFrameTime;
        void Start();
        void Stop();
    }
}