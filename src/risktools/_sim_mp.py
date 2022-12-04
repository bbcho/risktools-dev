
# def simOUcmp(s0, theta, mu, dt, sigma, T, sims=10, eps=None, max_processors=None):
#     ST = time.time()
#     fun = _import_csimOU()
#     N = int(T/dt)

#     if max_processors is None:
#         max_processors = int(mp.cpu_count())
    
#     pool = mp.Pool(max_processors)

#     mpsim = [int(sims/max_processors)]*max_processors
#     mpsim[-1] += (sims % mpsim[-1])

#     path_size = N + 1
    
#     try:
#         iter(mu)
#         if len(mu) != N+1:
#             raise ValueError("if mu is passed as an iterable, it must be of length int(T/dt) + 1 to account for starting value s0")
#     except:
#         mu = _np.ones((N+1))*mu

#     mu = _np.tile(_np.array(mu), sims)

#     if eps is None:
#         x = _np.random.normal(loc=0, scale=_np.sqrt(dt), size=((N+1)*sims))
#         x[0] = s0
#         x[::(N+1)] = s0
#     else:
#         x = eps.T
#         x = x.reshape((N+1)*sims)
#         x[:,0] = s0
    
#     print(time.time() - ST)
    
#     for i in range(max_processors):
#         s_idx = 0 if i == 0 else (path_size*mpsim[i-1]*i)
#         e_idx = (path_size*mpsim[i]*(i+1))

#         payload = [
#             x[s_idx:e_idx],
#             theta,
#             mu,
#             dt,
#             sigma,        
#             N+1,
#             mpsim[i]
#         ]

#         rr = pool.apply_async(
#             fun,
#             args = payload
#             # kwds=payload,
#         )

#     pool.close()
#     pool.join()
#     pool.terminate()
    
#     # fun(x, theta, mu, dt, sigma, N+1, sims)
    
#     return _pd.DataFrame(x.reshape((sims, N+1)).T)

# def simOUmp(s0=5, mu=4, theta=2, sigma=1, T=1, dt=1 / 252, sims=1000, max_processors=None):

#     result = Result()

#     if max_processors is None:
#         max_processors = int(mp.cpu_count())
#     pool = mp.Pool(max_processors)

#     mpsim = [int(sims/max_processors)]*max_processors

#     mpsim[-1] += (sims % mpsim[-1])

#     print(mpsim)

#     for i in range(max_processors):
#         print(time.time() - ST)
#         payload = dict(
#             s0=s0,
#             mu=mu,
#             theta=theta,
#             sigma=sigma,
#             T=T,
#             dt=dt,
#             sims=mpsim[i]
#         )

#         rr = pool.apply_async(
#             _simOUc,
#             #args=...
#             kwds=payload,
#             callback=result.update_result
#         )

#     pool.close()
#     pool.join()
#     pool.terminate()

#     return result.val#.T.reset_index(drop=True).T