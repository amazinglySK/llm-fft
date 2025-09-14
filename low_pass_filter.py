# FLinear

parser.add_argument("--fft_train_mode", type=int, default=0)
parser.add_argument("--fft_cut_freq", type=int, default=0)
parser.add_argument("--fft_base_T", type=int, default=24)
parser.add_argument("--fft_H_order", type=int, default=2)


if args.cut_freq == 0:
    args.cut_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10

self.model = Model(
    DotDict(
        {
            "seq_len": self.win_size // self.DSR,
            "enc_in": self.input_c,
            "individual": self.individual,
            "cut_freq": self.cutfreq,
            "pred_len": self.win_size - self.win_size // self.DSR,
        }
    )
)
self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def freq_dropout(self, x, y, dropout_rate=0.2, dim=0, keep_dominant=True):
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        xy = torch.cat([x,y],dim=0)
        xy_f = torch.fft.rfft(xy,dim=0)

        m = torch.FloatTensor(xy_f.shape).uniform_() < dropout_rate

        
        # amp = abs(xy_f)
        # _,index = amp.sort(dim=dim, descending=True)
        # dominant_mask = index > 5
        # m = torch.bitwise_and(m,dominant_mask)

        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=dim)

        x, y = xy[:x.shape[0],:].numpy(), xy[-y.shape[0]:,:].numpy()
        return x, y